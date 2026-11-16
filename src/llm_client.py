import os
from dotenv import load_dotenv
from gigachat import GigaChat
load_dotenv()
def ask_gigachat(prompt: str,temperature:float = 0.3) -> str:
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise ValueError("err: GIGACHAT_CREDENTIALS NOT IN .env")
    with GigaChat(credentials=credentials,verify_ssl_certs=False) as giga:
        response = giga.chat({
            "model" : "GigaChat-3-Lightning",
            "temperature" : temperature,
            "max_tokens" : 1000,
            "messages" : [
                {
                    "role" : "user",
                    "content" : prompt
                }
            ]
        })
        return response.choices[0].message.content
if __name__=="__main__":
    print("проверка связи с gigachat")
    test_question= "что такое промпт-инженеринг в трех предложениях?"
    try:
        answer= ask_gigachat(test_question)
        print(f"\n вопрос: {test_question}")
        print(f"ответ {answer}")
    except Exception as e:
        print(f"произошла ошибка при подлючении: {e}")