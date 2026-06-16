import re
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openai import OpenAI

SAMPLE_RATE = 16000

# VAD — detecção de fala por energia RMS
FRAME_MS        = 100
FRAME_SAMPLES   = int(SAMPLE_RATE * FRAME_MS / 1000)
VOZ_THRESHOLD   = 0.015   # RMS mínimo para considerar voz
PAUSA_SEGUNDOS  = 0.8     # silêncio contínuo → fim da fala
PAUSA_FRAMES    = int(PAUSA_SEGUNDOS * 1000 / FRAME_MS)
FALA_MIN_SEG    = 0.4     # descarta trechos muito curtos
FALA_MIN_FRAMES = int(FALA_MIN_SEG * 1000 / FRAME_MS)

# Dispositivos (macOS)
DEV_BLACKHOLE = 7   # BlackHole 2ch  — o que você ouve (sistema)
DEV_MIC       = 6   # Fifine Microphone — o que pessoas falam

cliente = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

print("Carregando Whisper (small)...")
whisper = WhisperModel("small", device="cpu", compute_type="int8")
print("Pronto.\n")

queue_sistema: queue.Queue = queue.Queue()
queue_mic: queue.Queue     = queue.Queue()


# ── filtros vindos do laguna ─────────────────────────────────────────────────

def is_hallucination(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 4:
        return False
    for n in (1, 2, 3, 4):
        if len(words) < n * 3:
            continue
        for i in range(len(words) - n * 3 + 1):
            block = words[i:i + n]
            if words[i + n:i + 2 * n] == block and words[i + 2 * n:i + 3 * n] == block:
                return True
    if len(words) < 8:
        return False
    if len(set(words)) / len(words) <= 0.3:
        return True
    for n in (2, 3, 4):
        if len(words) < n * 3:
            continue
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        counts: dict = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        top = max(counts.values())
        if top >= 3 and top / len(grams) > 0.3:
            return True
    return False


def limpar_tokens(text: str) -> str:
    return re.sub(r"[\[(][^)\]]*[\])]", "", text).strip()


# ── tradução via llama local ─────────────────────────────────────────────────

def traduzir(texto: str, idioma_origem: str) -> str:
    if idioma_origem == "en":
        origem, destino = "inglês", "português"
    else:
        origem, destino = "português", "inglês"

    resp = cliente.chat.completions.create(
        model="llama3.1:8b",
        messages=[{
            "role": "user",
            "content": (
                f"Traduza do {origem} para o {destino}. "
                f"Responda APENAS com a tradução, sem explicações, sem aspas:\n\n{texto}"
            ),
        }],
        temperature=0.3,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


# ── transcrição + tradução ───────────────────────────────────────────────────

def transcrever_e_traduzir(audio: np.ndarray, label: str) -> None:
    segments, info = whisper.transcribe(
        audio,
        beam_size=5,
        language=None,
        vad_filter=True,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.2,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    idioma = info.language
    texto  = limpar_tokens(" ".join(s.text for s in segments).strip())

    if not texto or is_hallucination(texto):
        return

    bandeira = "🇺🇸" if idioma == "en" else "🇧🇷" if idioma == "pt" else f"[{idioma}]"
    print(f"\n{label} {bandeira}  {texto}")

    if idioma in ("en", "pt"):
        traducao = traduzir(texto, idioma)
        dest_flag = "🇧🇷" if idioma == "en" else "🇺🇸"
        print(f"   {dest_flag}  {traducao}")

    print("─" * 60)


# ── VAD (máquina de estados por pausa) ──────────────────────────────────────

def processar(q: queue.Queue, label: str) -> None:
    estado          = "idle"
    frames_voz: list[np.ndarray] = []
    frames_silencio = 0
    raw_buffer      = np.array([], dtype=np.float32)

    while True:
        try:
            chunk      = q.get(timeout=0.5)
            raw_buffer = np.append(raw_buffer, chunk.flatten())

            while len(raw_buffer) >= FRAME_SAMPLES:
                frame      = raw_buffer[:FRAME_SAMPLES]
                raw_buffer = raw_buffer[FRAME_SAMPLES:]

                rms     = float(np.sqrt(np.mean(frame ** 2)))
                tem_voz = rms >= VOZ_THRESHOLD

                if estado == "idle":
                    if tem_voz:
                        estado          = "falando"
                        frames_voz      = [frame]
                        frames_silencio = 0
                        print(f"{label} falando...", end="\r", flush=True)

                elif estado == "falando":
                    frames_voz.append(frame)
                    if tem_voz:
                        frames_silencio = 0
                    else:
                        frames_silencio += 1
                        if frames_silencio >= PAUSA_FRAMES:
                            duracao = len(frames_voz) * FRAME_MS / 1000
                            print(f"⏸  {label} pausa ({duracao:.1f}s)        ", flush=True)

                            if len(frames_voz) >= FALA_MIN_FRAMES:
                                audio = np.concatenate(frames_voz)
                                threading.Thread(
                                    target=transcrever_e_traduzir,
                                    args=(audio, label),
                                    daemon=True,
                                ).start()

                            frames_voz      = []
                            frames_silencio = 0
                            estado          = "idle"

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break


# ── callbacks de áudio ───────────────────────────────────────────────────────

def make_callback(q: queue.Queue):
    def callback(indata, frames, time_info, status):
        mono = indata.mean(axis=1, keepdims=True) if indata.ndim > 1 else indata
        q.put(mono.copy())
    return callback


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Dispositivos ativos:")
    print(f"  [{DEV_BLACKHOLE}] BlackHole 2ch      → 🔊 SISTEMA (o que você ouve)")
    print(f"  [{DEV_MIC}] Fifine Microphone  → 🎤 MIC     (o que pessoas falam)\n")

    threading.Thread(target=processar, args=(queue_sistema, "🔊"), daemon=True).start()
    threading.Thread(target=processar, args=(queue_mic,     "🎤"), daemon=True).start()

    n_bh  = min(sd.query_devices(DEV_BLACKHOLE)["max_input_channels"], 2)
    n_mic = sd.query_devices(DEV_MIC)["max_input_channels"]

    print("Capturando. Ctrl+C para parar.\n")
    print("─" * 60)

    try:
        with sd.InputStream(device=DEV_BLACKHOLE, channels=n_bh,  samplerate=SAMPLE_RATE,
                            callback=make_callback(queue_sistema), dtype=np.float32):
            with sd.InputStream(device=DEV_MIC,       channels=n_mic, samplerate=SAMPLE_RATE,
                                callback=make_callback(queue_mic),     dtype=np.float32):
                while True:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nParado.")


if __name__ == "__main__":
    main()
