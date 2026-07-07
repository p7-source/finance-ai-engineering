# Day 1 — LangGraph Multi-Agent
# Yahoo context: Email processing supervisor + workers

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()

# 1. DEFINE STATE — shared memory across all agents
class EmailState(TypedDict):
    email: str
    summary: str
    action_items: str
    final_output: str

# 2. DEFINE MODELS
supervisor = ChatAnthropic(
  
    model="claude-haiku-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

worker = ChatAnthropic(
    # model="claude-haiku-20240307",
    model="claude-haiku-4-5",

    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# 3. DEFINE NODES (agents)
def summarize_agent(state: EmailState) -> EmailState:
    print("Summarize agent running...")
    response = worker.invoke(
        f"Summarize this email in one sentence:\n{state['email']}"
    )
    state["summary"] = response.content
    return state

def action_agent(state: EmailState) -> EmailState:
    print("Action agent running...")
    response = worker.invoke(
        f"Extract action items from this email:\n{state['email']}"
    )
    state["action_items"] = response.content
    return state

def supervisor_agent(state: EmailState) -> EmailState:
    print("Supervisor combining results...")
    response = supervisor.invoke(
        f"Combine this summary and action items into a final output:\n"
        f"Summary: {state['summary']}\n"
        f"Actions: {state['action_items']}"
    )
    state["final_output"] = response.content
    return state

# 4. BUILD GRAPH
graph = StateGraph(EmailState)

graph.add_node("summarize", summarize_agent)
graph.add_node("actions", action_agent)
graph.add_node("supervisor", supervisor_agent)

graph.set_entry_point("summarize")
graph.add_edge("summarize", "actions")
graph.add_edge("actions", "supervisor")
graph.add_edge("supervisor", END)

app = graph.compile()

# 5. RUN IT
email = """
Hi team, the Q3 budget review is scheduled for Friday 
at 2pm in Room 4B. John please bring the financial reports. 
Sarah please prepare the sales deck. All managers must 
confirm attendance by Thursday EOD.
"""

result = app.invoke({"email": email, "summary": "", 
                     "action_items": "", "final_output": ""})

print("\n=== FINAL OUTPUT ===")
print(result["final_output"])