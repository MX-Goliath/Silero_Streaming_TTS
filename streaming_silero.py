import sys
import torch

# Импортируем новую функцию
from silero_ollama_stream import process_text_to_speech_stream 


print("[INFO] Загружаю Silero‑TTS (RU)…")
RU_MODEL, _ = torch.hub.load("snakers4/silero-models", "silero_tts", language="ru", speaker="v4_ru")
RU_MODEL.to("cpu") 
print("[INFO] Silero готова.")



def main():

    ollama_model = "mistral-small3.1:latest"
    ollama_url = "http://localhost:11434"
    ollama_system_prompt_ru = (
        "You are a Russian-speaking assistant. "
        "Answer simple questions briefly. Write all numbers and formulas only in words. "
        "Put a period at the end of all sentences. Do not use lists."
    )

    silero_speaker = "baya"
    silero_sample_rate = 24000 
    silero_put_accent = True
    silero_put_yo = True

    try:
        prompt = input("\n» Ваш запрос: ")
        if not prompt:
            print("[INFO] Пустой запрос, завершение работы.")
            return

        process_text_to_speech_stream(
            user_prompt=prompt,
            silero_model_object=RU_MODEL, # передаем загруженную модель
            ollama_model_name=ollama_model,
            ollama_base_url=ollama_url,
            ollama_system_prompt=ollama_system_prompt_ru,
            silero_speaker=silero_speaker,
            silero_sample_rate=silero_sample_rate,
            silero_put_accent=silero_put_accent,
            silero_put_yo=silero_put_yo,
        )

    except EOFError: # Обработка Ctrl+D
        print("\n[INFO] Получен EOF, завершение работы.")
    except KeyboardInterrupt:
        print("\n[Прервано пользователем]")
    except Exception as e:
        print(f"\n[Критическая ошибка в main]: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
