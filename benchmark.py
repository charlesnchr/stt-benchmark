#!/usr/bin/env python3
"""
STT Latency Benchmark
=====================
Providers: Groq, OpenAI, Azure OpenAI, Deepgram, Together, Google Gemini
Measures: Time to first token (TTFT) and end-to-end (E2E) latency.

For batch/non-streaming APIs, TTFT == E2E.
For streaming APIs (Deepgram WS, Gemini streaming), TTFT < E2E.

Set API keys as environment variables before running. See .env.example.
"""

import base64
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

AUDIO_FILE = Path(__file__).parent / "test_audio.wav"
NUM_RUNS = 7  # first run dropped as warmup


def require_env(name):
    val = os.environ.get(name)
    if not val:
        return None
    return val


# ── Audio generation ──────────────────────────────────────────────────────────


def generate_test_audio():
    """Generate a short test audio clip using macOS TTS or a fallback."""
    if AUDIO_FILE.exists():
        return

    text = "The quick brown fox jumps over the lazy dog. This is a benchmark test for speech to text latency."

    if sys.platform == "darwin":
        aiff = AUDIO_FILE.with_suffix(".aiff")
        subprocess.run(["say", "-o", str(aiff), text], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff), str(AUDIO_FILE)],
            check=True,
        )
        print(f"Generated {AUDIO_FILE} via macOS TTS")
    else:
        print(
            f"No test audio found at {AUDIO_FILE}.\n"
            "On macOS it would be auto-generated. On other platforms,\n"
            "place a short WAV file there manually (16-bit PCM, any sample rate).",
            file=sys.stderr,
        )
        sys.exit(1)


# ── Benchmark functions ───────────────────────────────────────────────────────
# Each returns (ttft_seconds, e2e_seconds, transcript_text)


def bench_groq(audio_bytes, client):
    key = require_env("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {key}"}
    t0 = time.perf_counter()
    resp = client.post(
        url,
        headers=headers,
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"model": "whisper-large-v3-turbo", "response_format": "json"},
        timeout=30,
    )
    e2e = time.perf_counter() - t0
    resp.raise_for_status()
    return e2e, e2e, resp.json()["text"]


def bench_openai(audio_bytes, client):
    key = require_env("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {key}"}
    t0 = time.perf_counter()
    resp = client.post(
        url,
        headers=headers,
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"model": "whisper-1", "response_format": "json"},
        timeout=30,
    )
    e2e = time.perf_counter() - t0
    resp.raise_for_status()
    return e2e, e2e, resp.json()["text"]


def bench_azure(audio_bytes, client):
    key = require_env("AZURE_API_KEY")
    endpoint = require_env("AZURE_ENDPOINT")
    deployment = os.environ.get("AZURE_WHISPER_DEPLOYMENT", "whisper")
    url = f"{endpoint}/openai/deployments/{deployment}/audio/transcriptions?api-version=2024-10-01-preview"
    headers = {"api-key": key}
    t0 = time.perf_counter()
    resp = client.post(
        url,
        headers=headers,
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"response_format": "json"},
        timeout=30,
    )
    e2e = time.perf_counter() - t0
    resp.raise_for_status()
    return e2e, e2e, resp.json()["text"]


def bench_deepgram_batch(audio_bytes, client):
    key = require_env("DEEPGRAM_API_KEY")
    url = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true"
    headers = {"Authorization": f"Token {key}", "Content-Type": "audio/wav"}
    t0 = time.perf_counter()
    resp = client.post(url, headers=headers, content=audio_bytes, timeout=30)
    e2e = time.perf_counter() - t0
    resp.raise_for_status()
    text = resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
    return e2e, e2e, text


def bench_deepgram_streaming(audio_bytes, _client=None):
    """Deepgram streaming websocket for true TTFT."""
    import websockets.sync.client as ws_client

    key = require_env("DEEPGRAM_API_KEY")
    url = (
        "wss://api.deepgram.com/v1/listen?"
        "model=nova-3&encoding=linear16&sample_rate=22050&channels=1&interim_results=true"
    )
    headers = {"Authorization": f"Token {key}"}
    pcm_data = audio_bytes[44:]  # strip WAV header

    ttft = None
    finals = []

    t0 = time.perf_counter()
    with ws_client.connect(url, additional_headers=headers) as ws:
        ws.send(pcm_data)
        ws.send(json.dumps({"type": "CloseStream"}))

        while True:
            try:
                msg = ws.recv(timeout=10)
                t_recv = time.perf_counter()
                data = json.loads(msg)
                if data.get("type") == "Results":
                    text = data["channel"]["alternatives"][0].get("transcript", "")
                    if text.strip() and ttft is None:
                        ttft = t_recv - t0
                    if data.get("is_final") and text.strip():
                        finals.append(text)
            except Exception:
                break

    e2e = time.perf_counter() - t0
    return ttft or e2e, e2e, " ".join(finals)


def bench_together(audio_bytes, client):
    key = require_env("TOGETHER_API_KEY")
    url = "https://api.together.xyz/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {key}"}
    t0 = time.perf_counter()
    resp = client.post(
        url,
        headers=headers,
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"model": "openai/whisper-large-v3", "response_format": "json"},
        timeout=30,
    )
    e2e = time.perf_counter() - t0
    resp.raise_for_status()
    return e2e, e2e, resp.json()["text"]


def _gemini_body(audio_b64: str) -> dict:
    return {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                    {
                        "text": "Transcribe this audio exactly. Output only the transcription, nothing else."
                    },
                ]
            }
        ],
        "generationConfig": {"temperature": 0},
    }


def bench_gemini_2_flash(audio_bytes, client):
    key = require_env("GEMINI_API_KEY")
    audio_b64 = base64.b64encode(audio_bytes).decode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
    t0 = time.perf_counter()
    resp = client.post(url, json=_gemini_body(audio_b64), timeout=30)
    e2e = time.perf_counter() - t0
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return e2e, e2e, text


def bench_gemini_2_flash_stream(audio_bytes, client):
    """Gemini 2.0 Flash with streaming for true TTFT."""
    key = require_env("GEMINI_API_KEY")
    audio_b64 = base64.b64encode(audio_bytes).decode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse&key={key}"
    ttft = None
    chunks = []

    t0 = time.perf_counter()
    with client.stream("POST", url, json=_gemini_body(audio_b64), timeout=30) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            t_recv = time.perf_counter()
            data = json.loads(line[6:])
            parts = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            for p in parts:
                if "text" in p:
                    if ttft is None:
                        ttft = t_recv - t0
                    chunks.append(p["text"])

    e2e = time.perf_counter() - t0
    return ttft or e2e, e2e, "".join(chunks)


def bench_gemini_25_flash(audio_bytes, client):
    key = require_env("GEMINI_API_KEY")
    audio_b64 = base64.b64encode(audio_bytes).decode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
    t0 = time.perf_counter()
    resp = client.post(url, json=_gemini_body(audio_b64), timeout=30)
    e2e = time.perf_counter() - t0
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return e2e, e2e, text


def bench_gemini_3_flash(audio_bytes, client):
    key = require_env("GEMINI_API_KEY")
    audio_b64 = base64.b64encode(audio_bytes).decode()
    body = _gemini_body(audio_b64)
    body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "MINIMAL"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={key}"
    t0 = time.perf_counter()
    resp = client.post(url, json=body, timeout=30)
    e2e = time.perf_counter() - t0
    resp.raise_for_status()
    parts = resp.json()["candidates"][0]["content"]["parts"]
    text_parts = [p["text"] for p in parts if "text" in p]
    return e2e, e2e, " ".join(text_parts)


# ── Runner ────────────────────────────────────────────────────────────────────


def run_provider(name, func, audio_bytes, client, delay=1.5, needs_no_client=False):
    results = []
    for i in range(NUM_RUNS):
        if i > 0:
            time.sleep(delay)
        try:
            if needs_no_client:
                ttft, e2e, text = func(audio_bytes)
            else:
                ttft, e2e, text = func(audio_bytes, client)
            results.append((ttft, e2e, text))
            tag = f"TTFT={ttft*1000:>7.0f}ms  E2E={e2e*1000:>7.0f}ms"
            if i == 0:
                print(f"  [{i+1}] {tag}  \"{text[:65]}\"")
            else:
                print(f"  [{i+1}] {tag}")
        except Exception as e:
            err = str(e)[:120]
            print(f"  [{i+1}] ERROR: {err}")
    return results


def compute_stats(values):
    if len(values) < 2:
        return None
    s = sorted(values)
    p95_idx = min(int(len(s) * 0.95), len(s) - 1)
    return statistics.median(s), min(s), max(s), s[p95_idx]


def main():
    generate_test_audio()
    audio_bytes = AUDIO_FILE.read_bytes()
    duration_s = (len(audio_bytes) - 44) / (22050 * 2)
    print(f"Audio: {AUDIO_FILE.name} ({len(audio_bytes)/1024:.1f} KB, ~{duration_s:.1f}s)")
    print(f"Runs: {NUM_RUNS} per provider (first dropped as warmup)\n")

    client = httpx.Client(timeout=30)
    all_results = {}

    # Each entry: (display_name, func, delay_between_runs, needs_no_client, has_streaming_ttft, env_key)
    providers = [
        ("Groq (v3-turbo)", bench_groq, 1.5, False, False, "GROQ_API_KEY"),
        ("OpenAI (whisper-1)", bench_openai, 2.0, False, False, "OPENAI_API_KEY"),
        ("Azure OpenAI (whisper)", bench_azure, 21.0, False, False, "AZURE_API_KEY"),
        ("Deepgram (nova-3)", bench_deepgram_batch, 1.5, False, False, "DEEPGRAM_API_KEY"),
        ("Deepgram (nova-3 WS)", bench_deepgram_streaming, 2.0, True, True, "DEEPGRAM_API_KEY"),
        ("Together (whisper-v3)", bench_together, 1.5, False, False, "TOGETHER_API_KEY"),
        ("Gemini 2.0 Flash", bench_gemini_2_flash, 1.0, False, False, "GEMINI_API_KEY"),
        ("Gemini 2.0 Flash SSE", bench_gemini_2_flash_stream, 1.0, False, True, "GEMINI_API_KEY"),
        ("Gemini 2.5 Flash", bench_gemini_25_flash, 1.0, False, False, "GEMINI_API_KEY"),
        ("Gemini 3 Flash", bench_gemini_3_flash, 1.0, False, False, "GEMINI_API_KEY"),
    ]

    for display, func, delay, no_client, has_ttft, env_key in providers:
        if not os.environ.get(env_key):
            print(f"=== {display} === SKIPPED (no {env_key})")
            print()
            continue

        # Azure also needs AZURE_ENDPOINT
        if func == bench_azure and not os.environ.get("AZURE_ENDPOINT"):
            print(f"=== {display} === SKIPPED (no AZURE_ENDPOINT)")
            print()
            continue

        print(f"=== {display} ===")
        results = run_provider(
            display, func, audio_bytes, client, delay=delay, needs_no_client=no_client
        )
        all_results[display] = (has_ttft, results)
        print()

    client.close()

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(
        f"{'Provider':<30} {'TTFT':>8} {'E2E':>8} {'E2E min':>8} {'E2E max':>8} {'E2E p95':>8}  {'Note'}"
    )
    print(
        f"{'':30} {'(ms)':>8} {'med ms':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8}"
    )
    print("-" * 95)

    ranking = []

    for display, (has_ttft, runs) in all_results.items():
        if len(runs) < 2:
            print(f"{display:<30} {'':>8} {'FAILED / INSUFFICIENT DATA':>40}")
            continue

        warm_runs = runs[1:]
        ttfts = [r[0] * 1000 for r in warm_runs]
        e2es = [r[1] * 1000 for r in warm_runs]

        s_e2e = compute_stats(e2es)
        s_ttft = compute_stats(ttfts) if has_ttft else None

        if s_e2e is None:
            print(f"{display:<30} {'':>8} {'INSUFFICIENT DATA':>40}")
            continue

        e_med, e_min, e_max, e_p95 = s_e2e

        if s_ttft and has_ttft:
            t_med = s_ttft[0]
            ttft_str = f"{t_med:>7.0f}"
            sort_key = t_med
        else:
            ttft_str = f"{'=E2E':>7}"
            sort_key = e_med

        print(
            f"{display:<30} {ttft_str:>8} {e_med:>8.0f} {e_min:>8.0f} {e_max:>8.0f} {e_p95:>8.0f}"
            f"  {'streaming' if has_ttft else 'batch'}"
        )
        ranking.append((sort_key, display, has_ttft))

    print("=" * 95)

    ranking.sort()
    print("\nRanking by effective latency (batch=E2E median, streaming=TTFT median):")
    for i, (ms, name, is_stream) in enumerate(ranking, 1):
        label = "TTFT" if is_stream else "E2E"
        print(f"  {i}. {name}: {ms:.0f}ms ({label})")

    print(
        f"\nNotes:\n"
        f"  - Audio: ~{duration_s:.1f}s, 22kHz 16-bit mono WAV ({len(audio_bytes)/1024:.0f} KB)\n"
        f"  - {NUM_RUNS} runs per provider, first dropped as warmup\n"
        f"  - Batch APIs: TTFT == E2E (complete transcript returned at once)\n"
        f"  - Streaming APIs: TTFT = time to first partial result\n"
        f"  - Providers with missing API keys are skipped automatically\n"
        f"  - All times include network RTT from this machine\n"
    )


if __name__ == "__main__":
    main()
