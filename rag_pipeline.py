# Advanced RAG Pipeline
# Yahoo context: Regulatory document Q&A
# When RAG beats fine-tuning: dynamic knowledge retrieval

import os
import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("finance_docs")

# Simulated regulatory documents
documents = [
    {
        "id": "doc1",
        "text": "Basel III requires banks to maintain a minimum Common Equity Tier 1 ratio of 4.5%. Banks must also hold a capital conservation buffer of 2.5%, bringing the total to 7%.",
        "source": "Basel III Guidelines"
    },
    {
        "id": "doc2", 
        "text": "GDPR Article 17 gives individuals the right to erasure, also known as the right to be forgotten. Organizations must delete personal data upon request within 30 days.",
        "source": "GDPR Regulations"
    },
    {
        "id": "doc3",
        "text": "KYC (Know Your Customer) regulations require financial institutions to verify customer identity before opening accounts. Documents required include government ID and proof of address.",
        "source": "KYC Policy"
    },
    {
        "id": "doc4",
        "text": "Anti-Money Laundering regulations require suspicious transactions above $10,000 to be reported to FinCEN within 30 days via a Suspicious Activity Report (SAR).",
        "source": "AML Guidelines"
    },
    {
        "id": "doc5",
        "text": "The Volcker Rule prohibits banks from engaging in proprietary trading and limits their investments in hedge funds and private equity funds to 3% of Tier 1 capital.",
        "source": "Volcker Rule"
    }
]

# Store documents in ChromaDB
print("Storing documents in ChromaDB...")
collection.add(
    documents=[doc["text"] for doc in documents],
    ids=[doc["id"] for doc in documents],
    metadatas=[{"source": doc["source"]} for doc in documents]
)
print(f"Stored {len(documents)} documents\n")

def rag_query(question: str, n_results: int = 2) -> dict:
    # Retrieve relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    
    context = "\n\n".join(results["documents"][0])
    sources = [m["source"] for m in results["metadatas"][0]]
    
    # Generate answer using Claude
    prompt = f"""You are a financial compliance expert.
Answer the question using ONLY the context provided.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer concisely and cite the source."""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "question": question,
        "answer": response.content[0].text,
        "sources": sources,
        "chunks_retrieved": len(results["documents"][0])
    }

def print_result(result: dict):
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Chunks retrieved: {result['chunks_retrieved']}")
    print("-" * 50)

# Test queries
print("=== RAG PIPELINE TEST ===\n")

result1 = rag_query("What is the minimum capital ratio required by Basel III?")
print_result(result1)

result2 = rag_query("How should suspicious transactions be reported?")
print_result(result2)

result3 = rag_query("What are the KYC requirements for opening a bank account?")
print_result(result3)