"""Latency and throughput benchmarking script.

Measures end-to-end inference latency under concurrent load.
"""

import asyncio
import time
import httpx
import numpy as np
import io
import soundfile as sf


async def run_benchmark(url: str = "http://localhost:8000/v1/analyze", requests_count: int = 20, concurrency: int = 4):
    print(f"Starting latency benchmark against {url} ({requests_count} requests, concurrency={concurrency})...")

    # Generate test WAV in memory
    sr = 16000
    t = np.linspace(0, 5.0, sr * 5, dtype=np.float32)
    wave = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    audio_bytes = buf.getvalue()

    semaphore = asyncio.Semaphore(concurrency)
    latencies = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        async def make_request():
            async with semaphore:
                t0 = time.perf_counter()
                files = {"file": ("test.wav", audio_bytes, "audio/wav")}
                res = await client.post(url, files=files)
                t1 = time.perf_counter()
                if res.status_code == 200:
                    latencies.append((t1 - t0) * 1000)
                else:
                    print(f"Error {res.status_code}: {res.text}")

        tasks = [make_request() for _ in range(requests_count)]
        await asyncio.gather(*tasks)

    if latencies:
        print(f"\n--- Benchmark Results ---")
        print(f"Total Requests: {len(latencies)}")
        print(f"P50 Latency:   {np.percentile(latencies, 50):.2f} ms")
        print(f"P95 Latency:   {np.percentile(latencies, 95):.2f} ms")
        print(f"P99 Latency:   {np.percentile(latencies, 99):.2f} ms")
        print(f"Mean Latency:  {np.mean(latencies):.2f} ms")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
