from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Enterprise RAG Assistant is running"}