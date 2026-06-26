# sussurro

Offline neural machine translation in C++ on top of [ggml](https://github.com/ggml-org/ggml),
specialised in Romance languages. Runs fully on-device: no server, no network at runtime.

## Status

The translation core is functionally complete (v0.1): encoder-decoder Transformer inference with
greedy decoding, exposed as a library (`sussurro_core`) and a small CLI. Voice modes
(speech-to-text via whisper.cpp, text-to-speech) and the cross-platform Tauri UI are on the roadmap.

## Build

    git submodule update --init --recursive      # fetches ggml
    cmake -B build
    cmake --build build -j

The first configure also fetches and builds SentencePiece via CMake FetchContent (network needed once).
If you cloned without submodules, run instead:

    git submodule add https://github.com/ggml-org/ggml third_party/ggml

## Convert a model

    pip install -r requirements.txt
    python scripts/convert.py --model Helsinki-NLP/opus-mt-it-fr --outfile models/opus-mt-it-fr.f32.gguf

The whole model (weights, hyper-parameters, and both SentencePiece tokenizers) is packed into a
single self-contained `.gguf`.

## Translate

    ./build/sussurro -m models/opus-mt-it-fr.f32.gguf -p "Ciao, come stai?"

## Layout

    CMakeLists.txt        build (ggml submodule + SentencePiece fetch)
    scripts/convert.py    OPUS-MT (Hugging Face) -> .gguf
    src/sussurro.h        public API
    src/model.cpp         gguf loader (weights + metadata + embedded tokenizer)
    src/tokenizer.cpp     SentencePiece encode / decode
    src/encoder.cpp       Transformer encoder graph
    src/decoder.cpp       decoder graph (causal + cross attention) + greedy loop
    src/main.cpp          CLI

## Roadmap

- v0.2  KV cache, quantization (q8 / q4), beam search, speed stats
- v0.3  text-to-speech output
- v0.4  speech-to-text input (whisper.cpp)
- v0.5  voice-to-voice (offline pocket interpreter)
- v0.6  more language pairs, register / glossary control, BLEU / chrF evaluation
- Tauri cross-platform UI (Liquid Glass)

## Licences

Code: MIT (placeholder, your call). Model weights keep their own licence — OPUS-MT is CC-BY 4.0.
