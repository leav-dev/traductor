# Traductor — App de traducción de voz en tiempo real para reuniones

Fork de [sussurro.cpp](https://github.com/whispem/sussurro.cpp) (MIT license).

## Descripción

App de escritorio (cross-platform) que funciona como intermediario en reuniones:
- Recibe audio de una reunión y lo traduce en tiempo real (inglés ↔ español)
- Si la persona que tiene la app instalada habla, se traduce automáticamente y genera audio con su voz clonada
- El audio generado se envía como micrófono virtual a la app de reunión (Zoom, Teams, etc.)

## Stack

- **Traducción**: Marian/OPUS-MT en ggml (78M params, ~88MB) — en/es
- **STT**: whisper.cpp
- **TTS**: Qwen3-TTS (voice cloning zero-shot) via llama.cpp
- **VAD**: Silero VAD (ONNX)
- **Desktop**: Tauri v2 (Rust + JS)
- **Audio I/O**: cpal (Rust)

## Estado

Proyecto base inicial. Los modelos de IA y submodules de código NO están incluidos (ver SETUP.md).

## Setup

Ver [SETUP.md](SETUP.md) para instrucciones completas de inicialización en un PC autorizado.

## Licencia

MIT (heredada de sussurro.cpp)
