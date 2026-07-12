# react_agent.py
# ReAct Agent — Reason + Act + Observe pattern
# Agent decides when to search, what to search, when to stop
# Yahoo context: agentic AI workloads on Mail Intelligence

import os
import json
import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ============================================================
# Setup Vector Store
# ============================================================

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("react_docs")

documents = [
    {"id": "doc1", "text": "Basel III requires banks to maintain a minimum Common Equity Tier 1 ratio of 4.5%. Banks must also hold a capital conservation buffer of 2.5%, bringing the total to 7%.", "source": "Basel III"},
    {"id": "doc2", "text": "GDPR Article 17 gives individuals the right to erasure. Organizations must delete personal data upon request within 30 days.", "source": "GDPR"},
    {"id": "doc3", "text": "KYC regulations require financial institutions to verify customer identity before opening accounts. Documents required include government ID and proof of address.", "source": "KYC Policy"},
    {"id": "doc4", "text": "AML regulations require suspicious transactions above $10,000 to be reported to FinCEN within 30 days via a Suspicious Activity Report.", "source": "AML Guidelines"},
    {"id": "doc5", "text": "The Volcker Rule prohibits banks from proprietary trading and limits investments in hedge funds to 3% of Tier 1 capital.", "source": "Volcker Rule"},
    {"id": "doc6", "text": "Stress testing requires banks to assess their capital adequacy under adverse economic scenarios. Results must be reported to regulators annually.", "source": "Stress Testing Guidelines"},
    {"id": "doc7", "text": "Liquidity Coverage Ratio (LCR) requires banks to hold sufficient high-quality liquid assets to survive a 30-day stress scenario.", "source": "LCR Guidelines"},
]

collection.add(
    documents=[d["text"] for d in documents],
    ids=[d["id"] for d in documents],
    metadatas=[{"source": d["source"]} for d in documents]
)

# ============================================================
# Tools Available to Agent
# ============================================================

def search_regulations(query: str) -> str:
    """Search regulatory documents"""
    results = collection.query(query_texts=[query], n_results=2)
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    
    output = ""
    for doc, source in zip(docs, sources):
        output += f"[{source}]: {doc}\n\n"
    return output.strip()

def calculate(expression: str) -> str:
    """Simple calculator for financial ratios"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Calculation error"

TOOLS = {
    "search_regulations": {
        "description": "Search regulatory documents for compliance information",
        "function": search_regulations
    },
    "calculate": {
        "description": "Calculate financial ratios and numbers",
        "function": calculate
    }
}

# ============================================================
# ReAct Agent
# ============================================================

REACT_SYSTEM_PROMPT = """You are a financial compliance ReAct agent.
You have access to these tools:

1. search_regulations(query) — Search regulatory documents
2. calculate(expression) — Calculate financial values

Use this EXACT format for every response:

Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [input to the tool]

When you have enough information to answer:
Thought: I now have enough information to answer
Final Answer: [your complete answer]

Rules:
- Always start with a Thought
- Use tools to gather information before answering
- You can use tools multiple times
- Only give Final Answer when you have enough context
"""

def run_react_agent(question: str, max_steps: int = 5):
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    
    messages = [{"role": "user", "content": question}]
    
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            system=REACT_SYSTEM_PROMPT,
            messages=messages
        )
        
        agent_output = response.content[0].text
        print(agent_output)
        
        # Check if agent has final answer
        if "Final Answer:" in agent_output:
            final_answer = agent_output.split("Final Answer:")[-1].strip()
            print(f"\n✅ FINAL ANSWER: {final_answer}")
            return final_answer
        
        # Parse action
        if "Action:" in agent_output and "Action Input:" in agent_output:
            action_line = [l for l in agent_output.split("\n") if l.startswith("Action:")][0]
            input_line = [l for l in agent_output.split("\n") if l.startswith("Action Input:")][0]
            
            tool_name = action_line.replace("Action:", "").strip()
            tool_input = input_line.replace("Action Input:", "").strip()
            
            # Execute tool
            if tool_name in TOOLS:
                observation = TOOLS[tool_name]["function"](tool_input)
                print(f"\nObservation: {observation}")
                
                # Add to conversation
                messages.append({"role": "assistant", "content": agent_output})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\nContinue your reasoning."
                })
            else:
                print(f"Unknown tool: {tool_name}")
                break
        else:
            print("No action found — agent may be stuck")
            break
    
    return "Max steps reached"

# ============================================================
# Test ReAct Agent
# ============================================================

print("=== REACT AGENT — Financial Compliance ===")
print("Agent reasons, searches, and iterates to find answers\n")

# Test 1 — Simple single search
run_react_agent(
    "What is the Basel III capital requirement?"
)

# Test 2 — Multi-step reasoning
run_react_agent(
    "A bank has $100 million in Tier 1 capital. What is the maximum they can invest in hedge funds under the Volcker Rule? Show the calculation."
)

# Test 3 — Multi-document reasoning
run_react_agent(
    "What are all the reporting requirements for financial institutions — covering both suspicious transactions and stress testing?"
)