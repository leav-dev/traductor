#!/usr/bin/env bash
# Voice-to-voice demo: Italian audio in -> English spoken out.
# Usage: scripts/loop.sh <audio.wav>   (16 kHz mono WAV)
WHISPER="${WHISPER:-third_party/whisper.cpp/models/ggml-small.bin}"
MODEL="${MODEL:-models/tc-it-en.gguf}"
VOICE="${VOICE:-models/vits-piper-en_US-ryan-medium}"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <audio.wav>"
    exit 1
fi

out=$(./build/sussurro-interpret -w "$WHISPER" -m "$MODEL" -a "$1" 2>/dev/null \
        | sed -n 's/^translation: //p')

if [ -z "$out" ]; then
    echo "no translation produced"
    exit 1
fi

echo "IT -> EN: $out"
./build/sussurro-speak -k "$VOICE" --play -t "$out"