# Speech-to-Text API Latency Benchmark

How fast can you go from audio to text? If you're building a real-time voice pipeline -- think voice assistants, live captioning, or dictation -- the answer matters a lot. A 200ms difference in your STT step compounds with every other piece of your stack.

This repo benchmarks the major speech-to-text APIs head to head, measuring end-to-end latency and time to first token on the same audio clip, from the same machine, back to back.

## The Results

Tested February 2026 from a residential US connection. ~5.7s audio clip (macOS TTS, 22kHz 16-bit mono WAV, 247 KB). 7 runs per provider, first dropped as warmup.

| Rank | Provider | Model | Median | Min | Max | Type |
|------|----------|-------|--------|-----|-----|------|
| 1 | **Groq** | whisper-large-v3-turbo | **637ms** | 530ms | 764ms | batch |
| 2 | **Deepgram** | nova-3 | **710ms** | 557ms | 1551ms | batch |
| 3 | **Together** | whisper-large-v3 | **835ms** | 621ms | 1075ms | batch |
| 4 | Google Gemini | 2.0 Flash (SSE) | 1101ms | 984ms | 1278ms | streaming |
| 5 | Google Gemini | 2.0 Flash | 1149ms | 1036ms | 1584ms | batch |
| 6 | Azure OpenAI | whisper | 1275ms | 1238ms | 1325ms | batch |
| 7 | Deepgram | nova-3 (websocket) | 1359ms | 1353ms | 1392ms | streaming |
| 8 | OpenAI | whisper-1 | 1507ms | 1011ms | 1991ms | batch |
| 9 | Google Gemini | 2.5 Flash | 1673ms | 1404ms | 2269ms | batch |
| 10 | Google Gemini | 3 Flash (minimal) | 2300ms | 2047ms | 2413ms | batch |

### What stands out

**Groq wins on both speed and consistency.** A 637ms median with only ~230ms spread across runs. If you need one provider for real-time STT, this is it.

**Deepgram nova-3 is the runner-up** but watch the variance. Best case it's 557ms (faster than Groq's best), worst case 1551ms. You're rolling the dice on any given request.

**Together is a sleeper pick.** Running Whisper large-v3 at 835ms median with reasonable variance. Solid fallback if Groq goes down.

**Gemini models work for transcription but aren't built for it.** They're general-purpose LLMs processing audio multimodally, not dedicated STT engines. Gemini 2.0 Flash is the fastest of the bunch at ~1.1s. Gemini 2.5 and 3 Flash are progressively slower due to their reasoning overhead.

**Gemini 3 Flash can't turn off thinking.** Unlike 2.5 Flash (where `thinkingBudget: 0` works), Gemini 3 always thinks. The best you can do is `thinkingLevel: "MINIMAL"`, which still adds ~700ms over 2.0 Flash. This is a [known pain point](https://discuss.ai.google.dev/t/critical-feedback-mandatory-thinking-in-gemini-3-flash-is-a-regression-in-ux-and-cost-efficiency/116017).

**OpenAI and Azure are the slowest dedicated STT options.** Both >1.2s median. Azure also has aggressive rate limits (3 RPM on the default tier).

**Deepgram streaming was slower than batch here.** That's expected -- the websocket test sends all audio at once and waits. Streaming shines when you're sending audio live from a microphone and want partial results while the user is still talking. For buffered/recorded utterances, batch is faster.

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

Set API keys as environment variables. Any provider without a key is skipped automatically -- you don't need all of them.

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

- **E2E (end-to-end):** Time from sending the HTTP request to receiving the complete transcript. For batch APIs, this is the only meaningful metric.
- **TTFT (time to first token):** For streaming APIs (Deepgram websocket, Gemini SSE), time until the first partial transcript arrives. For batch APIs, TTFT equals E2E.
- **Warmup handling:** First run per provider is dropped from statistics to exclude cold-start and connection setup overhead.

All timing uses `time.perf_counter()` and includes full network round-trip time.

## Providers tested

| Provider | Model | API style | Notes |
|----------|-------|-----------|-------|
| [Groq](https://console.groq.com/) | whisper-large-v3-turbo | OpenAI-compatible batch | Fastest overall |
| [Deepgram](https://deepgram.com/) | nova-3 | REST batch + WebSocket streaming | Their own model, not Whisper |
| [Together](https://www.together.ai/) | whisper-large-v3 | OpenAI-compatible batch | Running full-size Whisper |
| [OpenAI](https://platform.openai.com/) | whisper-1 | REST batch | The original |
| [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) | whisper | REST batch | Requires a Whisper deployment |
| [Google Gemini](https://ai.google.dev/) | 2.0 Flash, 2.5 Flash, 3 Flash | REST batch + SSE streaming | LLM-based, not dedicated STT |

## Caveats

- **Location matters.** These numbers are from a US residential connection. Results will vary with geography and network conditions.
- **Audio length matters.** Longer audio will shift the rankings, particularly for providers that scale differently (Deepgram and Groq process audio faster than real-time; LLMs have fixed overhead per request).
- **Rate limits are real.** Azure's default Whisper deployment is limited to 3 RPM. OpenAI has per-minute quotas. Groq and Deepgram are more generous.
- **Gemini isn't a fair comparison.** It's an LLM doing transcription as a side effect of multimodal understanding. It's included because people ask about it, and because it actually works well for accuracy -- just not for speed.
- **This measures latency, not accuracy.** All providers transcribed the test audio correctly, but accuracy on noisy real-world audio is a different story. Deepgram nova-3 and Whisper large-v3 tend to lead on accuracy benchmarks.

## License

MIT
