from ollama import chat


class LLMService:

    def __init__(self, model="llama3.2:3b"):
        self.model = model

    def generate(self, prompt: str) -> str:

        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        return response.message.content.strip()