# Async FastAPI Service
# Yahoo context: serving layer behind yahoo.mlops CLI
# Wraps LLM router + RAG pipeline into production API

import os
import time
import asyncio
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

# Async client — non-blocking LLM calls
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# FastAPI app
app = FastAPI(
    title="Finance AI API",
    description="LLM routing + RAG for financial services",
    version="1.0.0"
)

# ChromaDB setup
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("finance_docs")

# Load documents
documents = [
    {"id": "doc1", "text": "Basel III requires banks to maintain a minimum Common Equity Tier 1 ratio of 4.5%. Banks must also hold a capital conservation buffer of 2.5%, bringing the total to 7%.", "source": "Basel III"},
    {"id": "doc2", "text": "GDPR Article 17 gives individuals the right to erasure. Organizations must delete personal data upon request within 30 days.", "source": "GDPR"},
    {"id": "doc3", "text": "KYC regulations require financial institutions to verify customer identity before opening accounts. Documents required include government ID and proof of address.", "source": "KYC Policy"},
    {"id": "doc4", "text": "AML regulations require suspicious transactions above $10,000 to be reported to FinCEN within 30 days via a Suspicious Activity Report.", "source": "AML Guidelines"},
    {"id": "doc5", "text": "The Volcker Rule prohibits banks from proprietary trading and limits investments in hedge funds to 3% of Tier 1 capital.", "source": "Volcker Rule"}
]

collection.add(
    documents=[d["text"] for d in documents],
    ids=[d["id"] for d in documents],
    metadatas=[{"source": d["source"]} for d in documents]
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class EmailRequest(BaseModel):
    email: str

class QueryResponse(BaseModel):
    answer: str
    model_used: str
    latency_sec: float
    cost_usd: float
    sources: list = []

# Model costs
MODEL_COSTS = {
    "claude-haiku-4-5": 0.00025,
    "claude-sonnet-4-6": 0.003,
}

def select_model(query: str) -> str:
    word_count = len(query.split())
    has_complex = any(w in query.lower() for w in [
        "analyze", "compare", "explain", "evaluate", "difference"
    ])
    return "claude-sonnet-4-6" if word_count > 30 or has_complex else "claude-haiku-4-5"

# Routes
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Finance AI API"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    start = time.time()
    model = select_model(request.question)

    # Retrieve from RAG
    results = collection.query(
        query_texts=[request.question],
        n_results=2
    )
    context = "\n\n".join(results["documents"][0])
    sources = [m["source"] for m in results["metadatas"][0]]

    # Async LLM call
    response = await client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Answer using this context only:\n{context}\n\nQuestion: {request.question}"
        }]
    )

    latency = round(time.time() - start, 2)
    tokens = response.usage.input_tokens
    cost = round((tokens / 1000) * MODEL_COSTS[model], 6)

    return QueryResponse(
        answer=response.content[0].text,
        model_used=model,
        latency_sec=latency,
        cost_usd=cost,
        sources=sources
    )

@app.post("/summarize", response_model=QueryResponse)
async def summarize_endpoint(request: EmailRequest):
    start = time.time()

    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"Summarize this email in one sentence and list action items:\n{request.email}"
        }]
    )

    latency = round(time.time() - start, 2)
    tokens = response.usage.input_tokens
    cost = round((tokens / 1000) * MODEL_COSTS["claude-haiku-4-5"], 6)

    return QueryResponse(
        answer=response.content[0].text,
        model_used="claude-haiku-4-5",
        latency_sec=latency,
        cost_usd=cost
    )

@app.post("/batch", response_model=list)
async def batch_endpoint(requests: list[QueryRequest]):
    # Async concurrent processing — key advantage
    tasks = [query_endpoint(req) for req in requests]
    results = await asyncio.gather(*tasks)
    return results