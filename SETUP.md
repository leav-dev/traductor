# Setup — Proyecto Traductor (fork de sussurro.cpp)

## Requisitos previos (PC autorizado)

- CMake 3.16+
- Compilador C++17 (GCC/Clang)
- Rust + Cargo (para Tauri)
- Node.js 18+ (para Tauri frontend)
- Python 3.10+ (para scripts de conversión de modelos)
- Git

## 1. Inicializar submodules (código de dependencias)

```bash
git submodule update --init --recursive
```

Esto descarga:
- `third_party/ggml` — tensor library (ggml-org/ggml)
- `third_party/whisper.cpp` — STT engine (ggml-org/whisper.cpp)
- `third_party/miniaudio` — audio playback (mackron/miniaudio)

## 2. Descargar sherpa-onnx (prebuilt, para TTS Piper)

Descargar desde: https://github.com/k2-fsa/sherpa-onnx/releases
Extraer en `third_party/sherpa-onnx/` con estructura:
```
third_party/sherpa-onnx/
├── include/sherpa-onnx/c-api/c-api.h
└── lib/
```

## 3. Descargar modelos de IA (PC autorizado únicamente)

### Modelos de traducción (Marian/OPUS-MT)
Descargar de HuggingFace y convertir a GGUF:
- `Helsinki-NLP/opus-mt-en-es` → `models/tc-en-es.gguf`
- `Helsinki-NLP/opus-mt-es-en` → `models/tc-es-en.gguf`
- `Helsinki-NLP/opus-mt-tc-big-en-es` → `models/tc-en-es.gguf` (mejor calidad)

### Modelos de STT (whisper.cpp)
Descargar de: https://huggingface.co/ggerganov/whisper.cpp
- `ggml-small.bin` → `third_party/whisper.cpp/models/ggml-small.bin`
- (o ggml-base.bin, ggml-tiny.bin para menor tamaño)

### Modelos de TTS (Qwen3-TTS para voice cloning)
Descargar de: https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF
- `qwen-talker-1.7b-base-Q8_0.gguf` → `models/qwen3-tts-1.7b-q8_0.gguf`
- `qwen-tokenizer-12hz-Q8_0.gguf` → `models/qwen-tokenizer-12hz-q8_0.gguf`

### VAD (Silero VAD)
Descargar de: https://github.com/snakers4/silero-vad
- `silero_vad.onnx` → `models/silero-vad.onnx`

## 4. Build

```bash
# Engine (C++)
cmake -B build && cmake --build build -j

# Desktop app (Tauri)
cd app && npm install && npm run tauri dev
```

## 5. Notas

- Los modelos de IA NO se incluyen en el repositorio (ver .gitignore)
- El proyecto base está listo para compilar una vez que los submodules y modelos estén presentes
- Los paths de modelos se configuran en `app/src-tauri/src/lib.rs`
