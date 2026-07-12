# ragas_eval.py
# RAG Pipeline Evaluation using RAGAS
# Measures: Faithfulness, Answer Relevancy, Context Precision
# Yahoo context: how we validated RAG quality before deployment

import os
import asyncio
from dotenv import load_dotenv
from anthropic import Anthropic
import chromadb

load_dotenv()

# ============================================================
# Setup
# ============================================================

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("finance_eval")

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

# ============================================================
# RAG Pipeline
# ============================================================

def retrieve_context(question):
    results = collection.query(query_texts=[question], n_results=2)
    context = results["documents"][0]
    return context

def generate_answer(question, contexts):
    context_text = "\n\n".join(contexts)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Answer using ONLY this context:\n{context_text}\n\nQuestion: {question}"
        }]
    )
    return response.content[0].text

# ============================================================
# RAGAS-style Evaluation
# ============================================================

def evaluate_faithfulness(answer, contexts):
    """
    Faithfulness: Is the answer grounded in the context?
    Score 0-1: 1 = fully faithful, 0 = hallucinated
    """
    context_text = "\n\n".join(contexts)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Rate faithfulness 0.0-1.0.
Is this answer ONLY based on the context provided?
1.0 = completely grounded in context
0.0 = contains information not in context

Context: {context_text}
Answer: {answer}

Reply with ONLY a number between 0.0 and 1.0"""
        }]
    )
    try:
        return float(response.content[0].text.strip())
    except:
        return 0.5

def evaluate_answer_relevancy(question, answer):
    """
    Answer Relevancy: Does the answer address the question?
    Score 0-1: 1 = perfectly relevant, 0 = irrelevant
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Rate answer relevancy 0.0-1.0.
Does this answer directly address the question?
1.0 = perfectly answers the question
0.0 = completely irrelevant

Question: {question}
Answer: {answer}

Reply with ONLY a number between 0.0 and 1.0"""
        }]
    )
    try:
        return float(response.content[0].text.strip())
    except:
        return 0.5

def evaluate_context_precision(question, contexts):
    """
    Context Precision: Are retrieved contexts relevant to the question?
    Score 0-1: 1 = perfectly relevant contexts, 0 = irrelevant
    """
    context_text = "\n\n".join(contexts)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Rate context precision 0.0-1.0.
Are these retrieved contexts relevant to answer the question?
1.0 = perfectly relevant contexts retrieved
0.0 = completely irrelevant contexts

Question: {question}
Contexts: {context_text}

Reply with ONLY a number between 0.0 and 1.0"""
        }]
    )
    try:
        return float(response.content[0].text.strip())
    except:
        return 0.5

# ============================================================
# Evaluation Dataset
# ============================================================

eval_dataset = [
    {
        "question": "What is the minimum capital ratio required by Basel III?",
        "ground_truth": "Basel III requires a minimum Common Equity Tier 1 ratio of 4.5% plus a 2.5% conservation buffer totaling 7%."
    },
    {
        "question": "What are KYC requirements for opening a bank account?",
        "ground_truth": "KYC requires identity verification with government ID and proof of address before opening accounts."
    },
    {
        "question": "How should suspicious transactions be reported?",
        "ground_truth": "Suspicious transactions above $10,000 must be reported to FinCEN within 30 days via a Suspicious Activity Report."
    },
    {
        "question": "What does the Volcker Rule prohibit?",
        "ground_truth": "The Volcker Rule prohibits banks from proprietary trading and limits hedge fund investments to 3% of Tier 1 capital."
    },
    {
        "question": "What is the GDPR right to erasure?",
        "ground_truth": "GDPR Article 17 gives individuals the right to have their personal data deleted within 30 days of request."
    }
]

# ============================================================
# Run Evaluation
# ============================================================

print("=== RAGAS EVALUATION — Finance AI RAG Pipeline ===\n")

results = []

for i, sample in enumerate(eval_dataset):
    print(f"Evaluating question {i+1}/{len(eval_dataset)}...")
    print(f"Q: {sample['question']}")

    # RAG pipeline
    contexts = retrieve_context(sample["question"])
    answer = generate_answer(sample["question"], contexts)

    # Evaluate
    faithfulness = evaluate_faithfulness(answer, contexts)
    relevancy = evaluate_answer_relevancy(sample["question"], answer)
    precision = evaluate_context_precision(sample["question"], contexts)

    result = {
        "question": sample["question"],
        "answer": answer,
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "context_precision": precision,
        "avg_score": round((faithfulness + relevancy + precision) / 3, 2)
    }
    results.append(result)

    print(f"A: {answer[:100]}...")
    print(f"📊 Faithfulness: {faithfulness} | Relevancy: {relevancy} | Precision: {precision}")
    print(f"⭐ Avg Score: {result['avg_score']}\n")

# ============================================================
# Final Report
# ============================================================

avg_faithfulness = round(sum(r["faithfulness"] for r in results) / len(results), 2)
avg_relevancy = round(sum(r["answer_relevancy"] for r in results) / len(results), 2)
avg_precision = round(sum(r["context_precision"] for r in results) / len(results), 2)
overall = round((avg_faithfulness + avg_relevancy + avg_precision) / 3, 2)

print("=" * 60)
print("=== FINAL RAGAS EVALUATION REPORT ===")
print("=" * 60)
print(f"Faithfulness Score:      {avg_faithfulness}")
print(f"Answer Relevancy Score:  {avg_relevancy}")
print(f"Context Precision Score: {avg_precision}")
print(f"Overall RAG Score:       {overall}")
print("=" * 60)

# Save results
import json
with open("ragas_results.json", "w") as f:
    json.dump({
        "summary": {
            "faithfulness": avg_faithfulness,
            "answer_relevancy": avg_relevancy,
            "context_precision": avg_precision,
            "overall": overall
        },
        "detailed_results": results
    }, f, indent=2)

print("\n✅ Results saved to ragas_results.json")
print("\nYour Interview Line:")
print(f"'My RAG pipeline achieves {overall} overall RAGAS score")
print(f" — {avg_faithfulness} faithfulness, {avg_relevancy} answer relevancy,")
print(f"   {avg_precision} context precision'")