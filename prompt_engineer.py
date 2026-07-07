# Prompt Engineering — Version control, A/B testing, quality measurement
# Yahoo context: consistent summary format across 225M users

import os
import time
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# PROMPT VERSIONS — version controlled like code
PROMPT_VERSIONS = {
    "v1": {
        "version": "v1",
        "description": "Basic prompt — no structure",
        "template": "Summarize this email: {email}"
    },
    "v2": {
        "version": "v2", 
        "description": "Structured prompt — with format instructions",
        "template": """You are an email intelligence assistant.
Summarize the email below in exactly this format:
SUMMARY: [one sentence]
ACTIONS: [bullet list of action items]
PRIORITY: [High/Medium/Low]

Email: {email}"""
    },
    "v3": {
        "version": "v3",
        "description": "Advanced prompt — with examples (few-shot)",
        "template": """You are an email intelligence assistant for a financial firm.

Example:
Email: "Hi John, please review the Q3 report by Friday and send feedback to Sarah."
SUMMARY: John must review Q3 report and send feedback to Sarah by Friday.
ACTIONS: 
- Review Q3 report
- Send feedback to Sarah by Friday
PRIORITY: High

Now process this email:
Email: {email}
SUMMARY:"""
    }
}

# Test emails
test_emails = [
    "Hi team, the budget meeting is rescheduled to Monday 2pm. CFO will present Q4 numbers. All department heads must attend and bring their expense reports.",
    "Reminder: Complete your compliance training by end of month or your system access will be suspended.",
    "FYI - the London office will be closed next week for renovation. Please coordinate with the NY team for any urgent matters."
]

def run_prompt(prompt_version: str, email: str) -> dict:
    prompt = PROMPT_VERSIONS[prompt_version]
    formatted = prompt["template"].format(email=email)
    
    start = time.time()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": formatted}]
    )
    latency = round(time.time() - start, 2)
    
    return {
        "version": prompt_version,
        "description": prompt["description"],
        "output": response.content[0].text,
        "tokens": response.usage.input_tokens + response.usage.output_tokens,
        "latency": latency,
        "cost": round((response.usage.input_tokens / 1000) * 0.00025, 6)
    }

def ab_test(email: str):
    print(f"\nEmail: {email[:60]}...")
    print("=" * 60)
    
    results = {}
    for version in PROMPT_VERSIONS:
        result = run_prompt(version, email)
        results[version] = result
        print(f"\n{version} — {result['description']}")
        print(f"Output: {result['output'][:200]}")
        print(f"Tokens: {result['tokens']} | Cost: ${result['cost']} | Latency: {result['latency']}s")
    
    # Pick winner
    best = min(results.values(), key=lambda x: x['tokens'])
    print(f"\n✅ Most efficient: {best['version']} ({best['tokens']} tokens)")
    return results

def save_results(all_results: list):
    with open("prompt_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n📊 Results saved to prompt_test_results.json")

# Run A/B test on all emails
print("=== PROMPT ENGINEERING A/B TEST ===")
print("Testing 3 prompt versions across 3 emails\n")

all_results = []
for email in test_emails:
    results = ab_test(email)
    all_results.append(results)

save_results(all_results)

# Summary
print("\n=== FINAL SUMMARY ===")
print("Version | Avg Tokens | Avg Cost | Avg Latency")
print("-" * 50)

for version in PROMPT_VERSIONS:
    avg_tokens = sum(r[version]["tokens"] for r in all_results) / len(all_results)
    avg_cost = sum(r[version]["cost"] for r in all_results) / len(all_results)
    avg_latency = sum(r[version]["latency"] for r in all_results) / len(all_results)
    print(f"{version}      | {round(avg_tokens)} tokens    | ${round(avg_cost, 6)} | {round(avg_latency, 2)}s")