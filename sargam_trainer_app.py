import streamlit as st
import numpy as np
import random
import io
import soundfile as sf

# ----------------------------
# Constants & Definitions
# ----------------------------

SAMPLE_RATE = 44100

WESTERN_SA_MAP = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
    "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88,
}

RAGA_DEF = {
    "Yaman": {"aaroh": ["N","R","G","M","D","N","S"], "avroh": ["S","N","D","P","M","G","R","S"]},
    "Bhoop": {"aaroh": ["S","R","G","P","D","S"], "avroh": ["S","D","P","G","R","S"]},
    "Bhairavi": {"aaroh": ["S","r","g","m","P","d","n","S"], "avroh": ["S","n","d","P","m","g","r","S"]},
    "Bhimpalasi": {"aaroh": ["S","g","m","P","n","S"], "avroh": ["S","n","D","P","m","g","R","S"]},
    "Durga": {"aaroh": ["S","R","m","P","D","S"], "avroh": ["S","D","P","m","R","S"]},
    "Khamaj": {"aaroh": ["S","R","G","m","P","D","N","S"], "avroh": ["S","n","D","P","m","G","R","S"]},
}

PAKAD_DEF = {
    "Yaman": [["N","R","G"],["N","R","M","G"],["M","P"],["M","D","P","N","D","P","M","R","G","R"],["N","R","D","N","S"]],
    "Bhoop": [["G","R","G"],["P","G"],["D","P"],["S","U","D","P","G"],["P","G","R","G"],["G","R","S"]],
    "Bhairavi": [["g","S","r","S"],["g","m","P"],["d","m","d","n","S","U"],["r","U","S","U","d","P","g","m","r","S"]],
    "Bhimpalasi": [["n","S","g","m","P"],["n","D","P"],["S","U"],["n","D","P"],["m","g","R","S"]],
    "Durga": [["m","P","D"],["m","R","D","S"],["R","R","P"]],
    "Khamaj": [["G","m","P","D"],["G","m","G"],["P","S","U","N","S","U"],["n","D","P"],["m","P","m","G"],["R","S"]],
}

# ----------------------------
# Harmonium Sound Engine
# ----------------------------

def harmonium_wave(freq, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    # Fundamental + harmonics (organ-like)
    wave = (
        1.0 * np.sin(2 * np.pi * freq * t) +
        0.6 * np.sin(2 * np.pi * 2 * freq * t) +
        0.3 * np.sin(2 * np.pi * 3 * freq * t) +
        0.15 * np.sin(2 * np.pi * 4 * freq * t)
    )

    # ADSR Envelope (harmonium feel)
    attack = int(0.08 * SAMPLE_RATE)
    release = int(0.15 * SAMPLE_RATE)
    sustain = len(wave) - attack - release

    envelope = np.concatenate([
        np.linspace(0, 1, attack),
        np.ones(sustain),
        np.linspace(1, 0, release)
    ])

    return wave * envelope

def build_sequence_wav_bytes(freqs, duration):
    audio = np.concatenate([harmonium_wave(f, duration) for f in freqs])

    # Normalize & soften
    audio /= np.max(np.abs(audio)) + 1e-6
    audio *= 0.6

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()

# ----------------------------
# Note Pools
# ----------------------------

def build_pool_free(sa_freq, octaves):
    ratios = {
        "S":1, "r":16/15, "R":9/8, "g":6/5, "G":5/4,
        "m":4/3, "M":45/32, "P":3/2, "d":8/5,
        "D":5/3, "n":9/5, "N":15/8,
    }
    pool = {}
    for o in octaves:
        mul = {"L":0.5,"M":1,"U":2}[o]
        for s,r in ratios.items():
            pool[s+o] = sa_freq * r * mul
    return pool

def build_pool_raga(sa_freq, raga, octaves):
    base = build_pool_free(sa_freq, ["M"])
    swaras = set(RAGA_DEF[raga]["aaroh"] + RAGA_DEF[raga]["avroh"])
    pool = {}
    for o in octaves:
        mul = {"L":0.5,"M":1,"U":2}[o]
        for s in swaras:
            pool[s+o] = base[s+"M"] * mul
    return pool

# ----------------------------
# UI
# ----------------------------

st.title("🎵 Sargam Trainer")

mode = st.radio("Mode", ["free","raga"], horizontal=True)
sa = st.selectbox("Base Sa", list(WESTERN_SA_MAP.keys()))
octaves = st.multiselect("Octaves", ["L","M","U"], default=["M"])
duration = st.slider("Note duration (seconds)", 0.3, 2.0, 0.8, 0.1)

if mode == "raga":
    raga = st.selectbox("Raga", list(RAGA_DEF.keys()))

st.header("🎧 Ear Tuning")

ear_mode = st.radio(
    "Play Mode",
    ["Play all notes","Play Aaroh","Play Avroh","Play Pakad (Mukhyan)"],
    horizontal=True
)

sa_freq = WESTERN_SA_MAP[sa]
pool = build_pool_free(sa_freq, octaves) if mode == "free" else build_pool_raga(sa_freq, raga, octaves)

if mode == "free" and ear_mode != "Play all notes":
    st.warning("Only 'Play all notes' is available in Free mode.")
    st.stop()

# Pakad Mode
if ear_mode == "Play Pakad (Mukhyan)" and mode == "raga":
    st.subheader("Pakad (Mukhyan)")
    for i, pakad in enumerate(PAKAD_DEF[raga]):
        label = " → ".join(pakad)
        if st.button(f"Play: {label}", key=f"pakad_{i}"):
            labels = [n+octaves[0] if len(n)==1 else n for n in pakad]
            freqs = [pool[l] for l in labels if l in pool]
            st.audio(build_sequence_wav_bytes(freqs, duration))
    st.stop()

if st.button("Play Ear Tuning"):
    if ear_mode == "Play all notes":
        labels = sorted(pool.keys(), key=lambda k: pool[k])
    elif ear_mode == "Play Aaroh":
        labels = [n+octaves[0] for n in RAGA_DEF[raga]["aaroh"]]
    elif ear_mode == "Play Avroh":
        labels = [n+octaves[0] for n in RAGA_DEF[raga]["avroh"]]

    freqs = [pool[l] for l in labels if l in pool]
    st.audio(build_sequence_wav_bytes(freqs, duration))
    st.write(" → ".join(labels))
