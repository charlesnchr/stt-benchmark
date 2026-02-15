# Speech-to-Text API Latency Benchmark

I wanted to know which STT API is actually fastest for real-time use. Not based on marketing pages or someone else's benchmark from last year, but measured myself, same audio, same machine, back to back.

This repo is the result. It tests Groq, OpenAI, Azure OpenAI, Deepgram, Together, and Google Gemini on a ~6 second audio clip and reports end-to-end latency and time to first token.

## Results

Tested February 2026 from a residential connection in London. The test audio is ~5.7s of macOS TTS (22kHz 16-bit mono WAV, 247 KB). Each provider was hit 7 times, with the first run dropped as warmup.

| Rank | Provider | Model | Median | Min | Max | Type |
|------|----------|-------|--------|-----|-----|------|
| 1 | Groq | whisper-large-v3-turbo | 637ms | 530ms | 764ms | batch |
| 2 | Deepgram | nova-3 | 710ms | 557ms | 1551ms | batch |
| 3 | Together | whisper-large-v3 | 835ms | 621ms | 1075ms | batch |
| 4 | Google Gemini | 2.0 Flash (SSE) | 1101ms | 984ms | 1278ms | streaming |
| 5 | Google Gemini | 2.0 Flash | 1149ms | 1036ms | 1584ms | batch |
| 6 | Azure OpenAI | whisper | 1275ms | 1238ms | 1325ms | batch |
| 7 | Deepgram | nova-3 (websocket) | 1359ms | 1353ms | 1392ms | streaming |
| 8 | OpenAI | whisper-1 | 1507ms | 1011ms | 1991ms | batch |
| 9 | Google Gemini | 2.5 Flash | 1673ms | 1404ms | 2269ms | batch |
| 10 | Google Gemini | 3 Flash (minimal) | 2300ms | 2047ms | 2413ms | batch |

### Observations

Groq is the fastest and most consistent. 637ms median with only ~230ms spread. If you're picking one provider for real-time STT, this is the obvious choice.

Deepgram nova-3 comes second but the variance is wide. Its best run (557ms) actually beat Groq's best, but its worst (1551ms) is nearly 3x the median. Any given request is unpredictable.

Together runs full Whisper large-v3 at 835ms median with less variance than Deepgram. Worth considering as a fallback.

The Gemini models can do transcription but they're general-purpose LLMs, not dedicated STT. Gemini 2.0 Flash is the fastest of the three at ~1.1s. The 2.5 and 3 Flash variants are slower because of their reasoning steps. Gemini 3 Flash is the worst here because you [can't fully disable its thinking](https://discuss.ai.google.dev/t/critical-feedback-mandatory-thinking-in-gemini-3-flash-is-a-regression-in-ux-and-cost-efficiency/116017). The closest option is `thinkingLevel: "MINIMAL"`, which still adds ~700ms compared to 2.0 Flash.

OpenAI and Azure are both above 1.2s. Azure also rate-limits aggressively at 3 RPM on the default Whisper tier.

Deepgram's websocket streaming was slower than its batch API in this test. That makes sense because the test sends all audio at once and waits. Streaming is designed for live microphone input where you want partial results while the user is still talking. For pre-recorded or buffered audio, batch wins.

## Running it yourself

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- macOS for auto-generated test audio, or provide your own `test_audio.wav`

### Setup

```bash
git clone https://github.com/charlesnchr/stt-benchmark.git
cd stt-benchmark

# Copy and fill in your API keys
cp .env.example .env
# Edit .env with your keys

# Run with uv (installs deps automatically)
uv run benchmark.py

# Or with pip
pip install httpx websockets
python benchmark.py
```

Set API keys as environment variables. Any provider without a key is skipped automatically, so you don't need all of them.

```bash
# Source your .env, or export individually:
export GROQ_API_KEY=gsk_...
export DEEPGRAM_API_KEY=...
export TOGETHER_API_KEY=...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...

# Azure (optional, needs a Whisper deployment)
export AZURE_API_KEY=...
export AZURE_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_WHISPER_DEPLOYMENT=whisper
```

On macOS, a test audio file is generated automatically using the `say` command. On other platforms, place a WAV file at `test_audio.wav` in the repo root.

### Example output

```
=== Groq (v3-turbo) ===
  [1] TTFT=    544ms  E2E=    544ms  "The quick brown fox jumps over the lazy dog..."
  [2] TTFT=    530ms  E2E=    530ms
  [3] TTFT=    746ms  E2E=    746ms
  ...

===============================================================================================
Provider                       TTFT      E2E  E2E min  E2E max  E2E p95  Note
                               (ms)   med ms     (ms)     (ms)     (ms)
-----------------------------------------------------------------------------------------------
Groq (v3-turbo)                =E2E      637      530      764      764  batch
Deepgram (nova-3)              =E2E      710      557     1551     1551  batch
Together (whisper-v3)          =E2E      835      621     1075     1075  batch
...

Ranking by effective latency (batch=E2E median, streaming=TTFT median):
  1. Groq (v3-turbo): 637ms (E2E)
  2. Deepgram (nova-3): 710ms (E2E)
  3. Together (whisper-v3): 835ms (E2E)
  ...
```

## What's being measured

E2E (end-to-end) is the time from sending the HTTP request to receiving the complete transcript. For batch APIs this is the only number that matters.

TTFT (time to first token) applies to streaming APIs (Deepgram websocket, Gemini SSE). It's the time until the first partial transcript arrives. For batch APIs, TTFT is the same as E2E.

The first run per provider is dropped to exclude cold-start and connection setup. All timing uses `time.perf_counter()` and includes network round-trip.

## Providers tested

| Provider | Model | API style | Notes |
|----------|-------|-----------|-------|
| [Groq](https://console.groq.com/) | whisper-large-v3-turbo | OpenAI-compatible batch | Fastest overall |
| [Deepgram](https://deepgram.com/) | nova-3 | REST batch + WebSocket streaming | Their own model, not Whisper |
| [Together](https://www.together.ai/) | whisper-large-v3 | OpenAI-compatible batch | Full-size Whisper |
| [OpenAI](https://platform.openai.com/) | whisper-1 | REST batch | The original |
| [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) | whisper | REST batch | Requires a deployed Whisper model |
| [Google Gemini](https://ai.google.dev/) | 2.0 Flash, 2.5 Flash, 3 Flash | REST batch + SSE streaming | LLM-based, not a dedicated STT model |

## Caveats

These numbers are from London over residential broadband. Your results will differ depending on where you are and what your network looks like.

Longer audio will change the rankings. Groq and Deepgram process audio faster than real-time, so longer clips don't proportionally increase latency. LLMs have more fixed overhead per request regardless of audio length.

Rate limits vary. Azure's default Whisper deployment allows 3 requests per minute. OpenAI has per-minute quotas. Groq and Deepgram are more generous.

Including Gemini is a bit unfair since it's an LLM doing transcription as a byproduct of multimodal understanding. It's here because people ask about it, and the accuracy is actually good. The speed just isn't there.

This only measures latency, not accuracy. Every provider transcribed the test audio correctly, but that's synthetic TTS audio. Performance on noisy real-world recordings is another question.

## License

MIT
