# sussurro

Offline neural machine translation **and voice** for English and the Romance languages. Speak or type in one language, read or hear it in another — across **English, Spanish, French and Italian**, fully on-device: no server, no network at runtime.

The translation engine is built from scratch on [ggml](https://github.com/ggml-org/ggml); speech recognition uses [whisper.cpp](https://github.com/ggml-org/whisper.cpp), and speech synthesis uses Piper voices via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (ONNX Runtime).

The name is Italian for *whisper*.

## Status — v0.7

A multilingual, offline **translator** with a native **desktop app**: choose a source and a target language, then type or speak, and read or hear the result — every stage is also usable from the command line.

- **Translate** — English, Spanish, French, Italian, in every direction. Encoder–decoder Transformers (OPUS-MT / Marian) reimplemented on ggml, with greedy and beam-search decoding, an incremental KV cache, sentence splitting, and q8_0 / q4_0 / f16 weights.
- **Listen** — multilingual speech-to-text via [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
- **Speak** — text-to-speech via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) running a Piper voice per language, played through [miniaudio](https://github.com/mackron/miniaudio).
- **Desktop app** — a [Tauri](https://v2.tauri.app) UI (Liquid Glass) wrapping all of the above.

Everything runs locally (Metal + Accelerate on Apple Silicon).

> **Tested on:** Apple Silicon (M5 Max), macOS Tahoe 26.4. Other Apple Silicon Macs and macOS versions should work but are untested. Intel Macs, Windows and Linux aren't verified yet — contributions welcome.

## Languages

Two kinds of model cover all twelve directions among **en / es / fr / it**, with no pivoting:

- a single **multilingual** model for Romance ↔ Romance, where the target language is chosen by a sentence-initial token (`>>fra<<`, `>>spa<<`, `>>ita<<`);
- **bilingual** models for English ↔ Romance (no token needed).

| From → To | Model | `-l` token |
|---|---|---|
| en → fr / es / it | `tc-en-fr` / `tc-en-es` / `tc-en-it` | — |
| fr → en | `tc-fr-en` | — |
| it → en | `tc-it-en` | — |
| es → en | `es-en` | — |
| it/es → fr | `tc-itc-itc` | `fra` |
| fr/es → it | `tc-itc-itc` | `ita` |
| fr/it → es | `tc-itc-itc` | `spa` |

`es → en` is the one pair with no `tc-big` model, so it uses the classic `opus-mt-es-en` instead — slightly older, but a high-resource pair, so quality holds.

The desktop app picks the right model and token automatically from the chosen languages; from the CLI you select them yourself.

## Components

- `sussurro_core` — translation library (model loading, SentencePiece tokenizer, encoder, decoder).
- `sussurro` — CLI: translate text (`-l <lang>` selects the target on multilingual models).
- `sussurro-quantize` — quantize a model to q8_0 / q4_0.
- `sussurro-interpret` — speech → text (whisper); add `-m` to also translate, or omit it to transcribe only.
- `sussurro-speak` — text → speech (WAV, and `--play` to play it).
- `scripts/loop.sh` — a simple voice-to-voice demo (Italian audio in → English spoken out).
- `app/` — the Tauri desktop application.

## Dependencies

- **ggml**, **whisper.cpp**, **miniaudio** — git submodules under `third_party/`.
- **SentencePiece** — fetched and built automatically by CMake (FetchContent, v0.2.0).
- **sherpa-onnx** — prebuilt C API library, downloaded manually; only needed for `sussurro-speak`.
- **Tauri v2** (Rust + Node), **cpal**, **hound** — for the desktop app in `app/`.

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

### Translation models

```bash
pip install -r requirements.txt

# Romance <-> Romance (target chosen at run time by the -l token)
python scripts/convert.py --model Helsinki-NLP/opus-mt-tc-big-itc-itc --outfile models/tc-itc-itc.gguf

# English <-> Romance (bilingual)
python scripts/convert.py --model Helsinki-NLP/opus-mt-tc-big-en-fr --outfile models/tc-en-fr.gguf
python scripts/convert.py --model Helsinki-NLP/opus-mt-tc-big-en-es --outfile models/tc-en-es.gguf
python scripts/convert.py --model Helsinki-NLP/opus-mt-tc-big-en-it --outfile models/tc-en-it.gguf
python scripts/convert.py --model Helsinki-NLP/opus-mt-tc-big-fr-en --outfile models/tc-fr-en.gguf
python scripts/convert.py --model Helsinki-NLP/opus-mt-tc-big-it-en --outfile models/tc-it-en.gguf

# Spanish -> English (classic OPUS-MT)
python scripts/convert.py --model Helsinki-NLP/opus-mt-es-en --outfile models/es-en.gguf
```

Each `.gguf` is self-contained (weights, hyper-parameters, and SentencePiece tokenizers), f16 by
default (add `--dtype f32` for full precision). To shrink them, quantize from an f32 export:

```bash
python scripts/convert.py --model Helsinki-NLP/opus-mt-tc-big-en-fr --outfile models/tc-en-fr.f32.gguf --dtype f32
./build/sussurro-quantize models/tc-en-fr.f32.gguf models/tc-en-fr.q8_0.gguf q8_0
```

### Speech-to-text model (whisper)

```bash
bash third_party/whisper.cpp/models/download-ggml-model.sh small
```

whisper is already multilingual; the source language is selected at run time (`-l`).

### Text-to-speech: sherpa-onnx + a voice per language

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

Then one Piper voice per output language:

```bash
cd models
for v in fr_FR-tom-medium en_US-ryan-medium es_ES-davefx-medium it_IT-paola-medium; do
  curl -SL -O "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-$v.tar.bz2"
  tar xf "vits-piper-$v.tar.bz2" && rm "vits-piper-$v.tar.bz2"
done
cd ..
```

Re-run `cmake -B build && cmake --build build -j` so `sussurro-speak` gets built.

## Command-line usage

Translate text (multilingual model needs a target token; bilingual models do not):

```bash
./build/sussurro -m models/tc-itc-itc.gguf -p "Ciao, come stai?" -l fra   # -> French
./build/sussurro -m models/tc-en-it.gguf   -p "Hello, how are you?"       # -> Italian
```

Transcribe speech (16 kHz mono WAV), optionally translating in the same pass:

```bash
./build/sussurro-interpret -w third_party/whisper.cpp/models/ggml-small.bin -a clip.wav -l es
./build/sussurro-interpret -w third_party/whisper.cpp/models/ggml-small.bin -a clip.wav -l it -m models/tc-it-en.gguf
```

Synthesize speech (and play it):

```bash
./build/sussurro-speak -k models/vits-piper-es_ES-davefx-medium -t "Hola, ¿cómo estás?" --play
```

Voice-to-voice demo — Italian audio in, English spoken out:

```bash
./scripts/loop.sh clip.wav
```

## Desktop app (Tauri)

`app/` is a desktop front-end built with [Tauri v2](https://v2.tauri.app) (Rust backend + a vanilla web UI) in the Liquid Glass interface: pick a source and target language, then speak or type, read the result, and hear it in that language's voice.
The swap button reverses the two languages.

Prerequisites: Rust 1.77+ and Node.js 20+ (plus Xcode Command Line Tools on macOS).

```bash
cd app
npm install
npm run tauri dev
```

The app calls the compiled engine binaries directly, so before running you need: the binaries built (`cmake --build build -j` at the repo root), the models and voices in place (above), and the `REPO` constant in `app/src-tauri/src/lib.rs` set to this repo's absolute path.
Microphone capture is native (via cpal); macOS asks for permission on first use (declared in `app/src-tauri/Info.plist`).

> **Note — early build.** The app shells out to the local engine binaries using an absolute path, so it
> runs on the machine where the repo lives; it is not yet a self-contained, shareable bundle. Bundling
> the engine, models and voices into the app and code-signing it are possible future work.

## Roadmap

- Windows and Linux support (developed on Apple Silicon; PRs very welcome).
- A self-contained, code-signed desktop bundle.
- More languages and pairs (Portuguese is one token away on the Romance model).
- A more expressive voice (e.g. Qwen3-TTS).
- Keyboard navigation in the app's language pickers.

## License & credits

sussurro's own source: released under the [MIT License](LICENSE).

Built on the work of others, each under its own license:

- Helsinki-NLP **OPUS-MT** models — CC-BY 4.0.
- **ggml** / **whisper.cpp** (ggml-org) — MIT.
- **SentencePiece** (Google) — Apache-2.0.
- **sherpa-onnx** (k2-fsa) — Apache-2.0.
- **miniaudio** (David Reid) — public domain / MIT-0.
- **Piper** voices (OHF-Voice / rhasspy) — see each voice's model card.
- **Tauri** (Tauri Programme / CommonsConservancy), **cpal**, **hound** — MIT / Apache-2.0.