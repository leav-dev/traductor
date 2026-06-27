# sussurro

Offline neural machine translation **and voice** for Romance languages, built from scratch on [ggml](https://github.com/ggml-org/ggml). Italian → French, fully on-device: no server, no network at runtime. 

The name is Italian for *whisper*.

## Status — v0.5

A working offline **voice-to-voice interpreter**: speak Italian, hear French — and every stage is also usable on its own.

- **Translate** — Italian text → French. Encoder–decoder Transformer (OPUS-MT / Marian architecture) reimplemented on ggml, with greedy and beam-search decoding, an incremental KV cache, sentence splitting, and q8_0 / q4_0 / f16 weights.
- **Listen** — Italian speech → text via [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
- **Speak** — French text → a French voice via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) running a Piper voice, played back through [miniaudio](https://github.com/mackron/miniaudio).
- **Loop** — `scripts/loop.sh` chains all three: Italian audio in → French spoken out.

Everything runs locally (Metal + Accelerate on Apple Silicon). The translation core is ggml; the text-to-speech leg uses ONNX (sherpa-onnx); audio playback is portable (miniaudio: CoreAudio /WASAPI / ALSA).

## Components

- `sussurro_core` — translation library (model loading, SentencePiece tokenizer, encoder, decoder).
- `sussurro` — CLI: translate Italian text → French.
- `sussurro-quantize` — quantize a model to q8_0 / q4_0.
- `sussurro-interpret` — Italian audio → (whisper) text → (sussurro) French text.
- `sussurro-speak` — French text → French speech (WAV, and `--play` to play it).
- `scripts/loop.sh` — voice-to-voice orchestrator (`interpret` → `speak --play`).

## Dependencies

- **ggml**, **whisper.cpp**, **miniaudio** — git submodules under `third_party/` (ggml is shared by the translation core and whisper.cpp; miniaudio is a single public-domain header for playback).
- **SentencePiece** — fetched and built automatically by CMake (FetchContent, v0.2.0).
- **sherpa-onnx** — prebuilt C API library, downloaded manually; only needed for `sussurro-speak`.

## Build

```bash
git clone --recurse-submodules https://github.com/whispem/sussurro.cpp.git
cd sussurro.cpp
cmake -B build && cmake --build build -j
```

The first build also fetches/builds SentencePiece and compiles whisper.cpp — a few minutes, once.
If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

`sussurro-speak` is built only once sherpa-onnx is present (see *Text-to-speech* below); the rest of the project builds without it.

## Models & voices

### Translation model

```bash
pip install -r requirements.txt
python scripts/convert.py --model Helsinki-NLP/opus-mt-it-fr --outfile models/opus-mt-it-fr.gguf
```

This packs the weights, hyper-parameters and both SentencePiece tokenizers into one self-contained `.gguf` (f16 by default; add `--dtype f32` for full precision). 
Quantization works from the f32 model:

```bash
python scripts/convert.py --model Helsinki-NLP/opus-mt-it-fr --outfile models/opus-mt-it-fr.f32.gguf --dtype f32
./build/sussurro-quantize models/opus-mt-it-fr.f32.gguf models/opus-mt-it-fr.q8_0.gguf q8_0
```

### Speech-to-text model (whisper)

```bash
bash third_party/whisper.cpp/models/download-ggml-model.sh small
```

Lands at `third_party/whisper.cpp/models/ggml-small.bin` (`base` is smaller/faster; `small` is better on Italian).

### Text-to-speech: sherpa-onnx + a French voice

Download the prebuilt sherpa-onnx C API library (macOS arm64 shown; pick your platform's `-shared` asset from the [releases](https://github.com/k2-fsa/sherpa-onnx/releases)):

```bash
cd third_party
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.3/sherpa-onnx-v1.13.3-osx-arm64-shared.tar.bz2
tar xf sherpa-onnx-v1.13.3-osx-arm64-shared.tar.bz2
mv sherpa-onnx-v1.13.3-osx-arm64-shared sherpa-onnx
rm sherpa-onnx-v1.13.3-osx-arm64-shared.tar.bz2
xattr -dr com.apple.quarantine sherpa-onnx   # macOS only: clear the download quarantine
cd ..
```

Then a French voice (Piper `tom`; other French voices: `siwis`, `upmc`, `tjiho-model1/2/3`):

```bash
cd models
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fr_FR-tom-medium.tar.bz2
tar xf vits-piper-fr_FR-tom-medium.tar.bz2
rm vits-piper-fr_FR-tom-medium.tar.bz2
cd ..
```

Re-run `cmake -B build && cmake --build build -j` so `sussurro-speak` gets built.

## Usage

Translate text:

```bash
./build/sussurro -m models/opus-mt-it-fr.gguf -p "Ho sempre pensato che tradurre fosse un'arte difficile."
# -> J'ai toujours pensé que traduire était un art difficile.
```

Flags: `-n <max tokens>` (default 256), `-b <beam size>` (default 1 = greedy).

Transcribe + translate Italian audio (16 kHz mono WAV):

```bash
./build/sussurro-interpret \
  -w third_party/whisper.cpp/models/ggml-small.bin \
  -m models/opus-mt-it-fr.q8_0.gguf \
  -a clip.wav
```

Convert any audio first with `ffmpeg -i in.m4a -ar 16000 -ac 1 -c:a pcm_s16le clip.wav`.

Synthesize French speech (and play it):

```bash
./build/sussurro-speak -k models/vits-piper-fr_FR-tom-medium -t "Salut, comment ça va ?" --play
```

Flags: `-o <file.wav>` (output, default `speak.wav`), `--play` (play through the speakers),
`-s <rate>` (>1 faster, <1 slower), `-v <speaker id>` (for multi-speaker voices).

Voice-to-voice — Italian audio in, French spoken out:

```bash
./scripts/loop.sh clip.wav
```

Override the defaults with env vars if needed, e.g. `VOICE=models/vits-piper-fr_FR-upmc-medium ./scripts/loop.sh clip.wav`.

## Pipeline

```
Italian speech ──whisper.cpp──▶ Italian text ──sussurro_core──▶ French text ──sherpa-onnx/Piper──▶ French speech ──miniaudio──▶ 🔊
            └────────── sussurro-interpret ──────────┘                        └──────────── sussurro-speak --play ───────────┘
                                          └──────────────────────── scripts/loop.sh ────────────────────────┘
```

## Roadmap

- A more expressive French voice (e.g. Qwen3-TTS) if neutral read-speech isn't enough.
- Cross-platform Tauri UI over `sussurro_core` via its C ABI.
- More Romance language pairs.
- Streaming / live microphone input.

## License & credits

sussurro's own source: released under the [MIT License](LICENSE).

Built on the work of others, each under its own license:

- Helsinki-NLP **OPUS-MT** models — CC-BY 4.0.
- **ggml** / **whisper.cpp** (ggml-org) — MIT.
- **SentencePiece** (Google) — Apache-2.0.
- **sherpa-onnx** (k2-fsa) — Apache-2.0.
- **miniaudio** (David Reid) — public domain / MIT-0.
- **Piper** voices (OHF-Voice / rhasspy) — see each voice's model card.