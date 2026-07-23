# ragas_eval_hybrid.py
# RAGAS Evaluation with Hybrid Search + Reranking
# Compares: Pure semantic (0.83) vs Hybrid + Reranking

import os
import json
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ============================================================
# Documents — defined FIRST before anything else
# ============================================================

documents = [
    {"id": "doc1", "text": "Basel III requires banks to maintain a minimum Common Equity Tier 1 ratio of 4.5%. Banks must also hold a capital conservation buffer of 2.5%, bringing the total to 7%.", "source": "Basel III"},
    {"id": "doc2", "text": "GDPR Article 17 gives individuals the right to erasure. Organizations must delete personal data upon request within 30 days.", "source": "GDPR"},
    {"id": "doc3", "text": "KYC regulations require financial institutions to verify customer identity before opening accounts. Documents required include government ID and proof of address.", "source": "KYC Policy"},
    {"id": "doc4", "text": "AML regulations require suspicious transactions above $10,000 to be reported to FinCEN within 30 days via a Suspicious Activity Report.", "source": "AML Guidelines"},
    {"id": "doc5", "text": "The Volcker Rule prohibits banks from proprietary trading and limits investments in hedge funds to 3% of Tier 1 capital.", "source": "Volcker Rule"}
]

texts = [d["text"] for d in documents]
sources = [d["source"] for d in documents]

# ============================================================
# Setup Hybrid Search Models
# ============================================================

print("Loading models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenized_docs = [doc.lower().split() for doc in texts]
bm25 = BM25Okapi(tokenized_docs)
doc_embeddings = embedder.encode(texts, normalize_embeddings=True)
print("✅ Models loaded\n")

# ============================================================
# Hybrid Search Functions
# ============================================================

def semantic_search(query, top_k=5):
    query_embedding = embedder.encode(query, normalize_embeddings=True)
    scores = np.dot(doc_embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(i, float(scores[i])) for i in top_indices]

def keyword_search(query, top_k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(i, float(scores[i])) for i in top_indices]

def hybrid_search(query, top_k=5, alpha=0.7):
    semantic_results = semantic_search(query, top_k)
    keyword_results = keyword_search(query, top_k)

    def normalize(results):
        if not results:
            return {}
        max_s = max(s for _, s in results)
        min_s = min(s for _, s in results)
        if max_s == min_s:
            return {i: 1.0 for i, _ in results}
        return {i: (s - min_s) / (max_s - min_s) for i, s in results}

    sem_norm = normalize(semantic_results)
    kw_norm = normalize(keyword_results)
    all_indices = set(sem_norm.keys()) | set(kw_norm.keys())
    combined = {i: alpha * sem_norm.get(i, 0) + (1 - alpha) * kw_norm.get(i, 0)
                for i in all_indices}
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

def rerank(query, candidates, top_k=2):
    pairs = [(query, texts[i]) for i, _ in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip([i for i, _ in candidates], scores),
                    key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# ============================================================
# RAG Pipeline with Hybrid + Reranking
# ============================================================

def retrieve_context(question):
    hybrid_results = hybrid_search(question, top_k=5)
    reranked = rerank(question, hybrid_results, top_k=2)
    return [texts[i] for i, _ in reranked]

def generate_answer(question, contexts):
    context_text = "\n\n".join(contexts)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role": "user",
                   "content": f"Answer using ONLY this context:\n{context_text}\n\nQuestion: {question}"}]
    )
    return response.content[0].text

# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_faithfulness(answer, contexts):
    context_text = "\n\n".join(contexts)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user",
                   "content": f"Rate 0.0-1.0: Is this answer ONLY from context?\nContext: {context_text}\nAnswer: {answer}\nReply with ONLY a number:"}]
    )
    try:
        return float(response.content[0].text.strip())
    except:
        return 0.5

def evaluate_answer_relevancy(question, answer):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user",
                   "content": f"Rate 0.0-1.0: Does this answer address the question?\nQuestion: {question}\nAnswer: {answer}\nReply with ONLY a number:"}]
    )
    try:
        return float(response.content[0].text.strip())
    except:
        return 0.5

def evaluate_context_precision(question, contexts):
    context_text = "\n\n".join(contexts)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user",
                   "content": f"Rate 0.0-1.0: Are these contexts relevant to the question?\nQuestion: {question}\nContexts: {context_text}\nReply with ONLY a number:"}]
    )
    try:
        return float(response.content[0].text.strip())
    except:
        return 0.5

# ============================================================
# Evaluation Dataset
# ============================================================

eval_dataset = [
    {"question": "What is the minimum capital ratio required by Basel III?"},
    {"question": "What are KYC requirements for opening a bank account?"},
    {"question": "How should suspicious transactions be reported?"},
    {"question": "What does the Volcker Rule prohibit?"},
    {"question": "What is the GDPR right to erasure?"}
]

# ============================================================
# Run Evaluation
# ============================================================

print("=== RAGAS EVALUATION — Hybrid Search + Reranking ===\n")

results = []

for i, sample in enumerate(eval_dataset):
    print(f"Evaluating {i+1}/5: {sample['question'][:50]}...")
    contexts = retrieve_context(sample["question"])
    answer = generate_answer(sample["question"], contexts)
    faithfulness = evaluate_faithfulness(answer, contexts)
    relevancy = evaluate_answer_relevancy(sample["question"], answer)
    precision = evaluate_context_precision(sample["question"], contexts)
    avg = round((faithfulness + relevancy + precision) / 3, 2)
    results.append({"faithfulness": faithfulness,
                    "answer_relevancy": relevancy,
                    "context_precision": precision,
                    "avg": avg})
    print(f"  Faithfulness: {faithfulness} | Relevancy: {relevancy} | Precision: {precision} | Avg: {avg}")

# ============================================================
# Final Report
# ============================================================

avg_f = round(sum(r["faithfulness"] for r in results) / len(results), 2)
avg_r = round(sum(r["answer_relevancy"] for r in results) / len(results), 2)
avg_p = round(sum(r["context_precision"] for r in results) / len(results), 2)
overall = round((avg_f + avg_r + avg_p) / 3, 2)

print(f"\n{'='*60}")
print(f"=== FINAL REPORT ===")
print(f"{'='*60}")
print(f"Faithfulness:       {avg_f}  (was 0.99)")
print(f"Answer Relevancy:   {avg_r}  (was 0.85)")
print(f"Context Precision:  {avg_p}  (was 0.66)")
print(f"Overall:            {overall}  (was 0.83)")
print(f"{'='*60}")
print(f"\nImprovement: 0.83 → {overall}")

with open("ragas_results_hybrid.json", "w") as f:
    json.dump({"summary": {"faithfulness": avg_f, "answer_relevancy": avg_r,
                           "context_precision": avg_p, "overall": overall},
               "vs_baseline": {"faithfulness": 0.99, "answer_relevancy": 0.85,
                               "context_precision": 0.66, "overall": 0.83}}, f, indent=2)

print("✅ Results saved to ragas_results_hybrid.json")
