#!/usr/bin/env bash
# Voice-to-voice: Italian audio in -> French spoken out.
# Usage: scripts/loop.sh <audio.wav>   (16 kHz mono WAV)
WHISPER="${WHISPER:-third_party/whisper.cpp/models/ggml-small.bin}"
MODEL="${MODEL:-models/opus-mt-it-fr.q8_0.gguf}"
VOICE="${VOICE:-models/vits-piper-fr_FR-tom-medium}"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <audio.wav>"
    exit 1
fi

fr=$(./build/sussurro-interpret -w "$WHISPER" -m "$MODEL" -a "$1" 2>/dev/null \
        | sed -n 's/^translation: //p')

if [ -z "$fr" ]; then
    echo "no translation produced"
    exit 1
fi

echo "IT -> FR: $fr"
./build/sussurro-speak -k "$VOICE" --play -t "$fr"