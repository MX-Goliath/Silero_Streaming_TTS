import json
import queue
import re
import sys
import threading
from typing import Generator

import numpy as np
import requests
import sounddevice as sd


SAFE_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё .,!?…:\\n-]")
MULTISPACES = re.compile(r"\\s{2,}")


def sanitize(text: str) -> str:
    text = SAFE_RE.sub(" ", text)
    text = MULTISPACES.sub(" ", text)
    return text.strip()




class StreamingSileroTTS:
    PUNCT = re.compile(r"[.!?…]")

    def __init__(self, silero_model, speaker: str, sample_rate: int, put_accent: bool, put_yo: bool):
        self.sr = sample_rate
        self.silero_model = silero_model
        self.speaker = speaker
        self.put_accent = put_accent
        self.put_yo = put_yo

        self.text_q: "queue.Queue[str | None]" = queue.Queue()
        self.audio_q: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._tts_thread = threading.Thread(target=self._tts_loop, daemon=True)
        self._stream_write_thread = threading.Thread(target=self._stream_write_loop, daemon=True)
        try:
            self.stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype='float32')
        except sd.PortAudioError as e:
             print(f"[ERROR] Не удалось создать аудио стрим: {e}", file=sys.stderr)
             raise

    def start(self):
        print("[DEBUG] Запуск аудио стрима...")
        self.stream.start()
        print("[DEBUG] Запуск TTS потока...")
        self._tts_thread.start()
        print("[DEBUG] Запуск потока записи в стрим...")
        self._stream_write_thread.start()

    def feed_text(self, chunk: str, final: bool = False):
        if chunk:
            self.text_q.put(chunk)
        if final:
            self.text_q.put(None)

    def wait(self):
        print("[DEBUG] Ожидание завершения TTS потока...")
        self._tts_thread.join()
        print("[DEBUG] Ожидание завершения потока записи в стрим...")
        self._stream_write_thread.join()
        print("[DEBUG] Остановка аудио стрима...")
        self.stream.stop()
        self.stream.close()
        print("[DEBUG] Ожидание завершено.")

    def _tts_loop(self):
        buf = ""
        while True:
            part = self.text_q.get()
            if part is None:
                if buf:
                    self._synthesize(buf)
                self.audio_q.put(None)
                break
            buf += part
            if self.PUNCT.search(part): # Если в текущем куске есть пунктуация
                self._synthesize(buf)
                buf = ""

    def _synthesize(self, raw: str):
        text = sanitize(raw)
        if not text:
            return

        try:
            wav = self.silero_model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=self.sr, # sample_rate модели Silero
                put_accent=self.put_accent,
                put_yo=self.put_yo,
            )
            wav_np = wav.cpu().numpy()
            self.audio_q.put(wav_np)
        except Exception as e:
            print(f"[ERROR] Ошибка во время TTS синтеза: {e}", file=sys.stderr)

    def _stream_write_loop(self):
        while True:
            wav = self.audio_q.get()
            if wav is None:
                print("[DEBUG] Получен сигнал завершения в audio_q, выход из цикла записи.")
                break
            if wav.size > 0:
                try:
                    self.stream.write(wav)
                except sd.PortAudioError as e:
                    print(f"[ERROR] Ошибка записи в аудио стрим: {e}", file=sys.stderr)
                    break
                except Exception as e:
                    print(f"[ERROR] Неизвестная ошибка при записи в стрим: {e}", file=sys.stderr)
                    break



def ollama_stream(
    prompt: str,
    model_name: str,
    base_url: str,
    system_prompt: str
) -> Generator[str, None, None]:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    with requests.post(url, json=payload, stream=True, timeout=3600) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            data = json.loads(raw)
            if data.get("done"):
                break
            yield data.get("message", {}).get("content", "")



def process_text_to_speech_stream(
    user_prompt: str,
    silero_model_object,
    ollama_model_name: str = "mistral:latest",
    ollama_base_url: str = "http://localhost:11434",
    ollama_system_prompt: str = (
        "Вы — русскоязычный ассистент. Отвечайте строго на русском языке. "
        "На простые вопросы отвечай кратко. Все цифры и формулы пиши только словами. "
        "В конце всех предложений ставь точку. Не используй списки."
    ),
    silero_speaker: str = "baya",
    silero_sample_rate: int = 48000, # Модели Silero v3/v4 обычно 48000 Hz
    silero_put_accent: bool = True,
    silero_put_yo: bool = True,
):
    """
    Обрабатывает запрос пользователя, получает потоковый ответ от Ollama
    и воспроизводит его с помощью Silero TTS.
    """
    tts = StreamingSileroTTS(
        silero_model=silero_model_object,
        speaker=silero_speaker,
        sample_rate=silero_sample_rate,
        put_accent=silero_put_accent,
        put_yo=silero_put_yo
    )
    tts.start()

    print("\\n[Ассистент]: ", end="", flush=True)
    try:
        for token in ollama_stream(
            prompt=user_prompt,
            model_name=ollama_model_name,
            base_url=ollama_base_url,
            system_prompt=ollama_system_prompt
        ):
            sys.stdout.write(token)
            sys.stdout.flush()
            tts.feed_text(token)
    except requests.exceptions.RequestException as e:
        print(f"\\n[ERROR] Ошибка подключения к Ollama: {e}", file=sys.stderr)
        tts.feed_text("", final=True)
        tts.wait()

        raise
    except Exception as e:
        print(f"\\n[ERROR] Ошибка во время стриминга от Ollama: {e}", file=sys.stderr)
        tts.feed_text("", final=True)
        tts.wait()
        raise 

    tts.feed_text("", final=True)
    tts.wait()
    print("\\n[INFO] Завершено.") 