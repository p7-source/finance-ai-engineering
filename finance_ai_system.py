# Finance AI System — Full Integration
# Connects: LangGraph + LLM Router + RAG + Monitoring
# Yahoo context: production ML platform serving 8 DS teams

import os
import time
import asyncio
import chromadb
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
from langsmith import traceable
from langsmith.wrappers import wrap_anthropic
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. SETUP — Models + DB + Monitoring
# ============================================================

raw_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client = wrap_anthropic(raw_client)  # LangSmith tracing

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("finance_docs_v2")

# Load regulatory documents
documents = [
    {"id": "doc1", "text": "Basel III requires minimum Common Equity Tier 1 ratio of 4.5% plus 2.5% conservation buffer = 7% total.", "source": "Basel III"},
    {"id": "doc2", "text": "GDPR Article 17 right to erasure — organizations must delete personal data within 30 days of request.", "source": "GDPR"},
    {"id": "doc3", "text": "KYC regulations require identity verification before opening accounts — government ID and proof of address required.", "source": "KYC Policy"},
    {"id": "doc4", "text": "AML regulations require suspicious transactions above $10,000 reported to FinCEN within 30 days via SAR.", "source": "AML Guidelines"},
    {"id": "doc5", "text": "Volcker Rule prohibits proprietary trading and limits hedge fund investments to 3% of Tier 1 capital.", "source": "Volcker Rule"}
]

collection.add(
    documents=[d["text"] for d in documents],
    ids=[d["id"] for d in documents],
    metadatas=[{"source": d["source"]} for d in documents]
)

# Model costs
MODEL_COSTS = {
    "claude-haiku-4-5": 0.00025,
    "claude-sonnet-4-6": 0.003,
}

# ============================================================
# 2. LLM ROUTER — Select model based on complexity
# ============================================================

def select_model(query: str) -> str:
    word_count = len(query.split())
    has_complex = any(w in query.lower() for w in [
        "analyze", "compare", "explain", "evaluate", "difference", "why"
    ])
    return "claude-sonnet-4-6" if word_count > 30 or has_complex else "claude-haiku-4-5"

# ============================================================
# 3. RAG RETRIEVAL — Get relevant context
# ============================================================

def retrieve_context(query: str) -> tuple:
    results = collection.query(query_texts=[query], n_results=2)
    context = "\n\n".join(results["documents"][0])
    sources = [m["source"] for m in results["metadatas"][0]]
    return context, sources

# ============================================================
# 4. LANGGRAPH STATE — Shared memory across agents
# ============================================================

class FinanceState(TypedDict):
    query: str
    query_type: str
    context: str
    sources: list
    model_used: str
    answer: str
    cost: float
    latency: float

# ============================================================
# 5. AGENTS — Specialized nodes
# ============================================================

def classifier_agent(state: FinanceState) -> FinanceState:
    """Classifies query as compliance, email, or general"""
    print(f"🔍 Classifying: {state['query'][:50]}...")
    
    compliance_keywords = ["basel", "gdpr", "kyc", "aml", "volcker", "capital", "regulation"]
    email_keywords = ["email", "meeting", "action", "summarize", "summary"]
    
    query_lower = state["query"].lower()
    
    if any(k in query_lower for k in compliance_keywords):
        state["query_type"] = "compliance"
    elif any(k in query_lower for k in email_keywords):
        state["query_type"] = "email"
    else:
        state["query_type"] = "general"
    
    print(f"📋 Query type: {state['query_type']}")
    return state

@traceable(name="compliance_agent", project_name="finance-prod")
def compliance_agent(state: FinanceState) -> FinanceState:
    """Handles regulatory compliance queries with RAG"""
    print(f"⚖️  Compliance agent processing...")
    
    context, sources = retrieve_context(state["query"])
    model = select_model(state["query"])
    
    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Answer using ONLY this regulatory context:\n{context}\n\nQuestion: {state['query']}"
        }]
    )
    
    latency = round(time.time() - start, 2)
    tokens = response.usage.input_tokens
    
    state["context"] = context
    state["sources"] = sources
    state["model_used"] = model
    state["answer"] = response.content[0].text
    state["cost"] = round((tokens / 1000) * MODEL_COSTS[model], 6)
    state["latency"] = latency
    
    return state

@traceable(name="email_agent", project_name="finance-prod")
def email_agent(state: FinanceState) -> FinanceState:
    """Handles email summarization"""
    print(f"📧 Email agent processing...")
    
    start = time.time()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Summarize in this format:
SUMMARY: [one sentence]
ACTIONS: [bullet list]
PRIORITY: [High/Medium/Low]

Email: {state['query']}"""
        }]
    )
    
    latency = round(time.time() - start, 2)
    tokens = response.usage.input_tokens
    
    state["model_used"] = "claude-haiku-4-5"
    state["answer"] = response.content[0].text
    state["cost"] = round((tokens / 1000) * MODEL_COSTS["claude-haiku-4-5"], 6)
    state["latency"] = latency
    state["sources"] = []
    
    return state

@traceable(name="general_agent", project_name="finance-prod")
def general_agent(state: FinanceState) -> FinanceState:
    """Handles general queries"""
    print(f"💬 General agent processing...")
    
    model = select_model(state["query"])
    
    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": state["query"]}]
    )
    
    latency = round(time.time() - start, 2)
    tokens = response.usage.input_tokens
    
    state["model_used"] = model
    state["answer"] = response.content[0].text
    state["cost"] = round((tokens / 1000) * MODEL_COSTS[model], 6)
    state["latency"] = latency
    state["sources"] = []
    
    return state

# ============================================================
# 6. ROUTING LOGIC
# ============================================================

def route_query(state: FinanceState) -> str:
    return state["query_type"]

# ============================================================
# 7. BUILD GRAPH
# ============================================================

graph = StateGraph(FinanceState)

graph.add_node("classifier", classifier_agent)
graph.add_node("compliance", compliance_agent)
graph.add_node("email", email_agent)
graph.add_node("general", general_agent)

graph.set_entry_point("classifier")
graph.add_conditional_edges(
    "classifier",
    route_query,
    {
        "compliance": "compliance",
        "email": "email",
        "general": "general"
    }
)
graph.add_edge("compliance", END)
graph.add_edge("email", END)
graph.add_edge("general", END)

app = graph.compile()

# ============================================================
# 8. RUN TESTS
# ============================================================

def run_query(query: str):
    print(f"\n{'='*60}")
    result = app.invoke({
        "query": query,
        "query_type": "",
        "context": "",
        "sources": [],
        "model_used": "",
        "answer": "",
        "cost": 0.0,
        "latency": 0.0
    })
    
    print(f"\n✅ ANSWER: {result['answer'][:300]}")
    print(f"🤖 Model: {result['model_used']}")
    print(f"💰 Cost: ${result['cost']}")
    print(f"⚡ Latency: {result['latency']}s")
    if result['sources']:
        print(f"📚 Sources: {result['sources']}")
    return result

# Test all 3 query types
print("=== FINANCE AI SYSTEM — FULL INTEGRATION TEST ===")

# Compliance query → compliance agent + RAG
run_query("What is the minimum capital ratio required by Basel III?")

# Email query → email agent
run_query("Hi team, budget meeting Monday 2pm. CFO presents Q4. All department heads must attend.")

# General query → general agent + LLM router
run_query("Explain the difference between Tier 1 and Tier 2 capital in simple terms.")