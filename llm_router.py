# LLM Router — Dynamic model selection based on query complexity
# Yahoo context: Qwen 1.7B for simple, larger model for complex

import os
import time
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Cost per 1K tokens (input)
MODEL_COSTS = {
    "claude-haiku-4-5":   0.00025,
    "claude-sonnet-4-6":  0.003,
}

def classify_query(query: str) -> str:
    word_count = len(query.split())
    has_complex_words = any(word in query.lower() for word in [
        "analyze", "compare", "explain", "design", "evaluate",
        "why", "how does", "difference between", "trade off"
    ])
    
    if word_count > 30 or has_complex_words:
        return "complex"
    return "simple"

def route_llm(query: str) -> dict:
    complexity = classify_query(query)
    
    if complexity == "simple":
        model = "claude-haiku-4-5"
    else:
        model = "claude-sonnet-4-6"
    
    print(f"\nQuery: {query[:50]}...")
    print(f"Complexity: {complexity}")
    print(f"Model selected: {model}")
    
    start_time = time.time()
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": query}]
        )
        
        latency = round(time.time() - start_time, 2)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = round((input_tokens / 1000) * MODEL_COSTS[model], 6)
        
        return {
            "model": model,
            "complexity": complexity,
            "response": response.content[0].text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "latency_sec": latency
        }
        
    except Exception as e:
        print(f"API failed: {e}")
        print("Falling back to local simulation...")
        return {
            "model": "fallback",
            "complexity": complexity,
            "response": "Fallback response — API unavailable",
            "cost_usd": 0,
            "latency_sec": 0
        }

def print_result(result: dict):
    print(f"\n=== RESULT ===")
    print(f"Model used:    {result['model']}")
    print(f"Latency:       {result['latency_sec']}s")
    print(f"Cost:          ${result['cost_usd']}")
    print(f"Input tokens:  {result.get('input_tokens', 'N/A')}")
    print(f"Response:      {result['response'][:200]}")
    print("="*40)

# Test 1 — Simple query → should route to Haiku
query1 = "What is a transformer model?"
result1 = route_llm(query1)
print_result(result1)

# Test 2 — Complex query → should route to Sonnet
query2 = """Analyze and compare the trade offs between 
fine-tuning a small LLM versus using RAG for a financial 
services document Q&A system. Consider latency, cost, 
accuracy and maintenance overhead."""
result2 = route_llm(query2)
print_result(result2)

# Test 3 — Cost comparison
print(f"\n=== COST COMPARISON ===")
print(f"Simple query cost:  ${result1['cost_usd']}")
print(f"Complex query cost: ${result2['cost_usd']}")
if result2['cost_usd'] > 0 and result1['cost_usd'] > 0:
    ratio = round(result2['cost_usd'] / result1['cost_usd'], 1)
    print(f"Sonnet costs {ratio}x more than Haiku")
print(f"Routing saves cost by using right model for right task")