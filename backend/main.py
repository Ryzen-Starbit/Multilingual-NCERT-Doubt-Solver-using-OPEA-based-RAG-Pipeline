from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.rag_pipeline import ask_question
from backend.chat_memory import add_message, get_conversation
from backend.feedback import save_feedback
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class Query(BaseModel):
    session_id: str
    question: str
    grade: str
    subject: str
class Feedback(BaseModel):
    question: str
    rating: int
    comment: str | None = None
@app.post("/ask")
def ask(query: Query):
    add_message(query.session_id, "user", query.question)
    result = ask_question(
        question=query.question,
        grade=query.grade,
        subject=query.subject
    )
    add_message(query.session_id, "assistant", result["answer"])
    result["history"] = get_conversation(query.session_id)
    return result
@app.post("/feedback")
def feedback(data: Feedback):
    save_feedback(data.question, data.rating, data.comment)
    return {"message": "Feedback saved"}

