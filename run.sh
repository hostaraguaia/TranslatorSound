#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

OLLAMA_STARTED=0

cleanup() {
    echo ""
    echo "Parando..."
    kill "$TRADUTOR_PID" 2>/dev/null || true
    if [ "$OLLAMA_STARTED" -eq 1 ]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup INT TERM

# Ollama running?
if ! pgrep -x ollama > /dev/null; then
    echo "Iniciando Ollama..."
    ollama serve &
    OLLAMA_PID=$!
    OLLAMA_STARTED=1
    sleep 3
fi

# Model pulled?
if ! ollama list | grep -q "llama3.1:8b"; then
    echo "Baixando llama3.1:8b..."
    ollama pull llama3.1:8b
fi

echo "Iniciando tradutor..."
uv run \
    --with numpy \
    --with sounddevice \
    --with faster-whisper \
    --with openai \
    python3 tradutor_tempo_real.py &
TRADUTOR_PID=$!
wait "$TRADUTOR_PID"
