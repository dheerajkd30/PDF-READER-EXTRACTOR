import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

app = FastAPI()

# Load vector DB
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="db", embedding_function=embedding)

# We'll store memories in this dict keyed by session_id
session_memories = {}

class ChatRequest(BaseModel):
    query: str
    session_id: str

@app.post("/chat")
async def chat(req: ChatRequest):
    # Retrieve or create memory for this session
    if req.session_id not in session_memories:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        session_memories[req.session_id] = memory
    else:
        memory = session_memories[req.session_id]

    # Create LLM chain with memory + retrieval
    llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

    response = qa_chain.run(req.query)
    return {"answer": response}