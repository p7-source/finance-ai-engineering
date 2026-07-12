# pinecone_rag.py
# Advanced RAG with Pinecone — cloud native vector store
# Replaces ChromaDB with production-grade Pinecone
# Shows vector DB breadth: ChromaDB (local) + Pinecone (cloud)

import os
import time
from dotenv import load_dotenv
from anthropic import Anthropic
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

# ============================================================
# Setup
# ============================================================

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Embedding model — same as ChromaDB uses internally
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded")

# Connect to existing index
index = pc.Index("finance-ai")
print("✅ Connected to Pinecone index: finance-ai")

# ============================================================
# Documents
# ============================================================

documents = [
    {"id": "doc1", "text": "Basel III requires banks to maintain a minimum Common Equity Tier 1 ratio of 4.5%. Banks must also hold a capital conservation buffer of 2.5%, bringing the total to 7%.", "source": "Basel III"},
    {"id": "doc2", "text": "GDPR Article 17 gives individuals the right to erasure. Organizations must delete personal data upon request within 30 days.", "source": "GDPR"},
    {"id": "doc3", "text": "KYC regulations require financial institutions to verify customer identity before opening accounts. Documents required include government ID and proof of address.", "source": "KYC Policy"},
    {"id": "doc4", "text": "AML regulations require suspicious transactions above $10,000 to be reported to FinCEN within 30 days via a Suspicious Activity Report.", "source": "AML Guidelines"},
    {"id": "doc5", "text": "The Volcker Rule prohibits banks from proprietary trading and limits investments in hedge funds to 3% of Tier 1 capital.", "source": "Volcker Rule"}
]

# ============================================================
# Ingest Documents
# ============================================================

print("\nIngesting documents into Pinecone...")

vectors = []
for doc in documents:
    embedding = embedder.encode(doc["text"]).tolist()
    vectors.append({
        "id": doc["id"],
        "values": embedding,
        "metadata": {
            "text": doc["text"],
            "source": doc["source"]
        }
    })

index.upsert(vectors=vectors)
print(f"✅ {len(vectors)} documents ingested into Pinecone")

# Wait for indexing
time.sleep(2)

# ============================================================
# Query Pipeline
# ============================================================

def query_pinecone(question: str, top_k: int = 2) -> dict:
    # Embed the question
    query_embedding = embedder.encode(question).tolist()
    
    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract context
    contexts = [match.metadata["text"] for match in results.matches]
    sources = [match.metadata["source"] for match in results.matches]
    scores = [round(match.score, 3) for match in results.matches]
    
    # Generate answer
    context_text = "\n\n".join(contexts)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Answer using ONLY this context:\n{context_text}\n\nQuestion: {question}"
        }]
    )
    
    return {
        "question": question,
        "answer": response.content[0].text,
        "sources": sources,
        "similarity_scores": scores,
        "contexts_retrieved": len(contexts)
    }

def print_result(result):
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer'][:200]}...")
    print(f"Sources: {result['sources']}")
    print(f"Similarity scores: {result['similarity_scores']}")
    print("-" * 50)

# ============================================================
# Test Queries
# ============================================================

print("\n=== PINECONE RAG PIPELINE TEST ===\n")

questions = [
    "What is the minimum capital ratio required by Basel III?",
    "What are KYC requirements for opening a bank account?",
    "How should suspicious transactions be reported?",
]

for question in questions:
    result = query_pinecone(question)
    print_result(result)

# ============================================================
# ChromaDB vs Pinecone Comparison
# ============================================================

print("\n=== CHROMADB vs PINECONE COMPARISON ===")
print("""
ChromaDB:
→ Local vector store
→ No network latency
→ Good for development
→ Not production scale
→ Free, no account needed

Pinecone:
→ Cloud native vector store
→ Managed, scalable to billions of vectors
→ Production ready
→ Real similarity scores returned
→ Enterprise features (namespaces, metadata filtering)
→ Used by: Notion, Shopify, Gong

Yahoo equivalent:
→ Vertex AI Vector Search (Google's managed vector store)
→ Same concept as Pinecone but on GCP
→ Scales to 1B+ vectors
→ Integrated with Vertex AI pipelines
""")

print("✅ Pinecone RAG pipeline complete")
print("Portfolio now shows: ChromaDB + Pinecone + Vertex AI knowledge")