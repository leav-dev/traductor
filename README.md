# sussurro

Offline neural machine translation **and voice** for Romance languages, built from scratch on [ggml](https://github.com/ggml-org/ggml). 
Italian → French, fully on-device: no server, no network at runtime. 

The name is Italian for *whisper*.

## Status — v0.6

An offline **voice-to-voice interpreter** with a native **desktop app**: speak Italian (or type it), read the French, hear it spoken — and every stage is also usable from the command line.

- **Translate** — Italian text → French. Encoder–decoder Transformer (OPUS-MT / Marian architecture) reimplemented on ggml, with greedy and beam-search decoding, an incremental KV cache, sentence splitting, and q8_0 / q4_0 / f16 weights.
- **Listen** — Italian speech → text via [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
- **Speak** — French text → a French voice via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) running a Piper voice, played back through [miniaudio](https://github.com/mackron/miniaudio).
- **Desktop app** — a [Tauri](https://v2.tauri.app) UI (Liquid Glass) wrapping all of the above.

Everything runs locally (Metal + Accelerate on Apple Silicon).

## Components

- `sussurro_core` — translation library (model loading, SentencePiece tokenizer, encoder, decoder).
- `sussurro` — CLI: translate Italian text → French.
- `sussurro-quantize` — quantize a model to q8_0 / q4_0.
- `sussurro-interpret` — Italian audio → (whisper) text → (sussurro) French text.
- `sussurro-speak` — French text → French speech (WAV, and `--play` to play it).
- `scripts/loop.sh` — voice-to-voice orchestrator (`interpret` → `speak --play`).
- `app/` — the Tauri desktop application.

## Dependencies

- **ggml**, **whisper.cpp**, **miniaudio** — git submodules under `third_party/`.
- **SentencePiece** — fetched and built automatically by CMake (FetchContent, v0.2.0).
- **sherpa-onnx** — prebuilt C API library, downloaded manually; only needed for `sussurro-speak`.
- **Tauri v2** (Rust + Node) — for the desktop app in `app/` (see below).

## Build (engine)

```bash
git clone --recurse-submodules https://github.com/whispem/sussurro.cpp.git
cd sussurro.cpp
cmake -B build && cmake --build build -j
```

The first build also fetches/builds SentencePiece and compiles whisper.cpp — a few minutes, once.
If you cloned without `--recurse-submodules`: `git submodule update --init --recursive`.

`sussurro-speak` is built only once sherpa-onnx is present (see *Text-to-speech* below).

## Models & voices

### Translation model

```bash
pip install -r requirements.txt
python scripts/convert.py --model Helsinki-NLP/opus-mt-it-fr --outfile models/opus-mt-it-fr.gguf
```

Packs the weights, hyper-parameters and both SentencePiece tokenizers into one self-contained `.gguf`
(f16 by default; add `--dtype f32` for full precision). Quantization works from the f32 model:

```bash
python scripts/convert.py --model Helsinki-NLP/opus-mt-it-fr --outfile models/opus-mt-it-fr.f32.gguf --dtype f32
./build/sussurro-quantize models/opus-mt-it-fr.f32.gguf models/opus-mt-it-fr.q8_0.gguf q8_0
```

### Speech-to-text model (whisper)

```bash
bash third_party/whisper.cpp/models/download-ggml-model.sh small
```

### Text-to-speech: sherpa-onnx + a French voice

Download the prebuilt sherpa-onnx C API library (macOS arm64 shown; pick your platform's `-shared` asset from the [releases](https://github.com/k2-fsa/sherpa-onnx/releases)):

```bash
cd third_party
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.3/sherpa-onnx-v1.13.3-osx-arm64-shared.tar.bz2
tar xf sherpa-onnx-v1.13.3-osx-arm64-shared.tar.bz2
mv sherpa-onnx-v1.13.3-osx-arm64-shared sherpa-onnx
rm sherpa-onnx-v1.13.3-osx-arm64-shared.tar.bz2
xattr -dr com.apple.quarantine sherpa-onnx   # macOS only
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

## Command-line usage

Translate text:

```bash
./build/sussurro -m models/opus-mt-it-fr.gguf -p "Ho sempre pensato che tradurre fosse un'arte difficile."
```

Transcribe + translate Italian audio (16 kHz mono WAV):

```bash
./build/sussurro-interpret -w third_party/whisper.cpp/models/ggml-small.bin -m models/opus-mt-it-fr.q8_0.gguf -a clip.wav
```

Synthesize French speech (and play it):

```bash
./build/sussurro-speak -k models/vits-piper-fr_FR-tom-medium -t "Salut, comment ça va ?" --play
```

Voice-to-voice — Italian audio in, French spoken out:

```bash
./scripts/loop.sh clip.wav
```

## Desktop app (Tauri)

`app/` is a desktop front-end built with [Tauri v2](https://v2.tauri.app) (Rust backend + a vanilla web UI) in the Liquid Glass interface: speak Italian or type it, read the French, hear it spoken.

Prerequisites: Rust 1.77+ and Node.js 20+ (plus Xcode Command Line Tools on macOS).

```bash
cd app
npm install
npm run tauri dev
```

The app calls the compiled engine binaries directly, so before running you need: the binaries built (`cmake --build build -j` at the repo root), the models and French voice in place (above), and the `REPO` constant in `app/src-tauri/src/lib.rs` set to this repo's absolute path. 
Microphone capture is native (via cpal); macOS asks for permission on first use (declared in `app/src-tauri/Info.plist`).

Build a macOS bundle: `npm run tauri build` → an unsigned `.app` / `.dmg` under `app/src-tauri/target/release/bundle/`.

> **Note — early build.** The app shells out to the local engine binaries via an absolute path, so it
> runs on the machine where the repo lives; it is not yet a self-contained, shareable bundle. Bundling
> the engine and models into the app (Tauri sidecars + resources) and code-signing/notarization are
> planned for v0.7.

## Roadmap

- **v0.7 — multilingual**: English and Spanish, plus a working language swap, via OPUS-MT multilingual models with target-language tokens. whisper already transcribes these; each output language needs its Piper voice.
- A self-contained, code-signed desktop bundle.
- A more expressive voice (e.g. Qwen3-TTS).
- More language pairs.

## License & credits

sussurro's own source: released under the [MIT License](LICENSE).

Built on the work of others, each under its own license:

- Helsinki-NLP **OPUS-MT** models — CC-BY 4.0.
- **ggml** / **whisper.cpp** (ggml-org) — MIT.
- **SentencePiece** (Google) — Apache-2.0.
- **sherpa-onnx** (k2-fsa) — Apache-2.0.
- **miniaudio** (David Reid) — public domain / MIT-0.
- **Piper** voices (OHF-Voice / rhasspy) — see each voice's model card.
- **Tauri** (Tauri Programme / CommonsConservancy) — MIT / Apache-2.0.