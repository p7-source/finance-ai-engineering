# traffic_generator.py
# Synthetic traffic generator
# Simulates real user load hitting the API

import asyncio
import aiohttp
import time
import random

API_URL = "http://127.0.0.1:8000"

QUERIES = [
    {"question": "What is Basel III?"},
    {"question": "What are KYC requirements?"},
    {"question": "How to report suspicious transactions?"},
    {"question": "What is the Volcker Rule?"},
    # {"question": "Analyze and compare the difference between Basel III capital requirements and AML regulations and their impact on financial institutions"},
    {"question": "What is GDPR"},
]

async def send_query(session, request_id):
    query = random.choice(QUERIES)
    start = time.time()
    try:
        async with session.post(
            f"{API_URL}/query",
            json=query,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            data = await response.json()
            latency = round(time.time() - start, 2)
            print(f"✅ Request {request_id} | {latency}s | model: {data.get('model_used', 'N/A')}")
            return latency
    except Exception as e:
        latency = round(time.time() - start, 2)
        print(f"❌ Request {request_id} | {latency}s | Error: {str(e)[:50]}")
        return None

async def check_metrics(session):
    async with session.get(f"{API_URL}/metrics") as r:
        metrics = await r.json()
        print(f"\n📊 Metrics: {metrics}\n")

async def traffic_wave(requests_per_second, duration_seconds):
    print(f"\n🌊 Wave: {requests_per_second} req/sec for {duration_seconds} seconds")
    print("=" * 60)
    async with aiohttp.ClientSession() as session:
        request_id = 0
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            tasks = []
            for _ in range(requests_per_second):
                request_id += 1
                tasks.append(send_query(session, request_id))
            await asyncio.gather(*tasks)
            await check_metrics(session)
            await asyncio.sleep(1)

async def main():
    print("=== SYNTHETIC TRAFFIC GENERATOR ===")
    print("Simulating real user load patterns\n")

    print("Phase 1: Low traffic (2 req/sec)")
    await traffic_wave(requests_per_second=2, duration_seconds=15)

    print("\nPhase 2: Medium traffic (5 req/sec)")
    await traffic_wave(requests_per_second=5, duration_seconds=15)

    print("\nPhase 3: Traffic SPIKE (10 req/sec)")
    await traffic_wave(requests_per_second=10, duration_seconds=15)

    print("\nPhase 4: Back to normal (2 req/sec)")
    await traffic_wave(requests_per_second=2, duration_seconds=15)

    print("\n=== TRAFFIC GENERATION COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(main())