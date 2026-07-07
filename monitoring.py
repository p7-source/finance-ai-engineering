# LangSmith Monitoring — Level 4 observability
# Traces every LLM call, tracks quality, alerts on degradation

import os
import time
from anthropic import Anthropic
from langsmith import Client, traceable
from langsmith.wrappers import wrap_anthropic
from dotenv import load_dotenv

load_dotenv()

# Wrap Anthropic client with LangSmith tracing
raw_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client = wrap_anthropic(raw_client)

ls_client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))

# Prompt versions
PROMPTS = {
    "v1": "Summarize this email: {email}",
    "v2": """Summarize the email in this exact format:
SUMMARY: [one sentence]
ACTIONS: [bullet list]
PRIORITY: [High/Medium/Low]

Email: {email}"""
}

@traceable(name="email_summarization", project_name="finance-prod")
def summarize_email(email: str, prompt_version: str = "v2") -> dict:
    prompt = PROMPTS[prompt_version].format(email=email)
    
    start = time.time()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = round(time.time() - start, 2)
    
    output = response.content[0].text
    
    # Quality check — does output follow format?
    has_summary = "SUMMARY:" in output
    has_actions = "ACTIONS:" in output
    has_priority = "PRIORITY:" in output
    quality_score = sum([has_summary, has_actions, has_priority]) / 3
    
    return {
        "output": output,
        "prompt_version": prompt_version,
        "latency": latency,
        "tokens": response.usage.input_tokens + response.usage.output_tokens,
        "quality_score": quality_score,
        "format_check": {
            "has_summary": has_summary,
            "has_actions": has_actions,
            "has_priority": has_priority
        }
    }

# Test emails
emails = [
    "Hi team, budget meeting Monday 2pm. CFO presents Q4. All department heads must attend with expense reports.",
    "Complete compliance training by month end or system access suspended.",
    "London office closed next week for renovation. Coordinate with NY team for urgent matters."
]

print("=== LANGSMITH MONITORING TEST ===")
print("All calls will be traced in LangSmith dashboard\n")

for i, email in enumerate(emails):
    print(f"Email {i+1}: {email[:50]}...")
    
    # Test both versions
    result_v1 = summarize_email(email, "v1")
    result_v2 = summarize_email(email, "v2")
    
    print(f"v1 quality score: {result_v1['quality_score']}")
    print(f"v2 quality score: {result_v2['quality_score']}")
    print(f"v2 wins: {result_v2['quality_score'] > result_v1['quality_score']}")
    print("-" * 40)

print("\n✅ Check your LangSmith dashboard:")
print("https://smith.langchain.com")
print("Project: finance-prod")
print("You should see all traces logged there")