from app.generator import LLMService


llm = LLMService()


response = llm.generate(
    "Explain what an embedding is in one sentence."
)


print(response)