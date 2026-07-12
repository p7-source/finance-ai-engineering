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
from collections import deque

load_dotenv()

# Async client — non-blocking LLM calls
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# FastAPI app
app = FastAPI(
    title="Finance AI API",
    description="LLM routing + RAG for financial services",
    version="1.0.0"
)

# ============================================================
# CLOSED LOOP CONCURRENCY CONTROL
# Yahoo context: telemetry-driven autoscaling
# ============================================================

MAX_CONCURRENT = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT)
request_times = deque(maxlen=100)
queue_depth = 0

def get_p99_latency():

    now = time.time()
    # Only look at last 30 seconds
    recent = [t for ts, t in request_times if now - ts < 30]
    
    if len(recent) < 5:
        return 0
    sorted_times = sorted(recent)
    return sorted_times[int(len(sorted_times) * 0.99)]
    # if len(request_times) < 10:
    #     return 0
    # sorted_times = sorted(request_times)
    # return sorted_times[int(len(sorted_times) * 0.99)]

async def closed_loop_controller():
    global semaphore, MAX_CONCURRENT
    while True:
        await asyncio.sleep(3)
        p99 = get_p99_latency()
        current = MAX_CONCURRENT

        if p99 > 5.0:
            MAX_CONCURRENT = max(1, MAX_CONCURRENT - 2)
            semaphore = asyncio.Semaphore(MAX_CONCURRENT)
            print(f"⬇️  Decreasing: {current} → {MAX_CONCURRENT} (p99={p99:.2f}s)")

        elif p99 < 2.0 and MAX_CONCURRENT < 20:
            MAX_CONCURRENT = MAX_CONCURRENT + 2
            semaphore = asyncio.Semaphore(MAX_CONCURRENT)
            print(f"⬆️  Increasing: {current} → {MAX_CONCURRENT} (p99={p99:.2f}s)")

        else:
            print(f"✅ Stable: MAX_CONCURRENT={MAX_CONCURRENT} p99={p99:.2f}s")

@app.on_event("startup")
async def startup():
    asyncio.create_task(closed_loop_controller())

@app.get("/metrics")
async def metrics():
    return {
        "max_concurrent": MAX_CONCURRENT,
        "p99_latency_sec": get_p99_latency(),
        "queue_depth": queue_depth,
        "requests_tracked": len(request_times)
    }

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
    global queue_depth
    queue_depth += 1
    
    async with semaphore:
        queue_depth -= 1
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
        # request_times.append(latency)
        request_times.append((time.time(), latency))
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