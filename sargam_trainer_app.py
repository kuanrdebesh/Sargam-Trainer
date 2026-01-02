import streamlit as st
import numpy as np
import sounddevice as sd
import random

# ==========================================
# Globals
# ==========================================
SAMPLE_RATE = 44100
VOLUME = 0.2

WESTERN_SA_MAP = {
    "C": 261.63, "C#": 277.18, "DB": 277.18,
    "D": 293.66, "D#": 311.13, "EB": 311.13,
    "E": 329.63,
    "F": 349.23, "F#": 369.99, "GB": 369.99,
    "G": 392.00, "G#": 415.30, "AB": 415.30,
    "A": 440.00, "A#": 466.16, "BB": 466.16,
    "B": 493.88,
}

SWARA_OFFSETS = [
    ("S",0),("r",1),("R",2),("g",3),
    ("G",4),("m",5),("M",6),
    ("P",7),("d",8),("D",9),
    ("n",10),("N",11)
]
SWARA_TO_OFFSET = {s:o for s,o in SWARA_OFFSETS}

OCTAVE_FACTORS = {"L":0.5,"M":1.0,"U":2.0}
OCTAVE_SUFFIX  = {"L":",","M":"","U":"'"}

RAGA_DEF = {
    "Yaman":{
        "swaras":["S","R","G","M","P","D","N"],
        "aaroh":["N","R","G","M","P","D","N","S"],
        "avroh":["S","N","D","P","M","G","R","S"]
    },
    "Bhairavi":{
        "swaras":["S","r","g","m","P","d","n"],
        "aaroh":["S","r","g","m","P","d","n","S"],
        "avroh":["S","n","d","P","m","g","r","S"]
    },
    "Bhoop":{
        "swaras":["S","R","G","P","D"],
        "aaroh":["S","R","G","P","D","S"],
        "avroh":["S","D","P","G","R","S"]
    },
    "Bhimpalasi":{
        "swaras":["S","R","g","m","P","D","n","N"],
        "aaroh":["n","S","g","m","P","N","S"],
        "avroh":["S","N","D","P","m","g","R","S"]
    },
    "Durga":{
        "swaras":["S","R","m","P","D"],
        "aaroh":["S","R","m","P","D","S"],
        "avroh":["S","D","P","m","R","S"]
    },
    "Khamaj":{
        "swaras":["S","R","G","m","P","D","N","n"],
        "aaroh":["S","G","M","P","D","N","S"],
        "avroh":["S","n","D","P","M","G","R","S"]
    }
}

# ==========================================
# Audio
# ==========================================

def play_tone(freq, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = np.sin(2 * np.pi * freq * t) * VOLUME
    sd.play(tone, SAMPLE_RATE)
    sd.wait()


# ==========================================
# Sequence Generation
# ==========================================

def build_pool_free(sa_freq, komal, octaves):
    swaras = [s for s,_ in SWARA_OFFSETS] if komal else ["S","R","G","m","P","D","N"]
    pool = {}
    for o in octaves:
        for s in swaras:
            sym = s + OCTAVE_SUFFIX[o]
            pool[sym] = sa_freq * OCTAVE_FACTORS[o] * (2 ** (SWARA_TO_OFFSET[s] / 12))
    return pool

def build_pool_raga(sa_freq, raga, octaves):
    swaras = RAGA_DEF[raga]["swaras"]
    pool = {}
    for o in octaves:
        for s in swaras:
            sym = s + OCTAVE_SUFFIX[o]
            pool[sym] = sa_freq * OCTAVE_FACTORS[o] * (2 ** (SWARA_TO_OFFSET[s] / 12))
    return pool

def generate_free(pool, count):
    keys = list(pool.keys())
    return [random.choice(keys) for _ in range(count)]

def generate_raga(pool, count, raga, pattern):
    if pattern == "free":
        return generate_free(pool, count)

    order = RAGA_DEF[raga]["aaroh"] if pattern == "aaroh" else RAGA_DEF[raga]["avroh"]

    mapping = {}
    for sym in pool:
        base = sym.rstrip("',")
        mapping.setdefault(base, []).append(sym)

    valid = [s for s in order if s in mapping]
    rank = {s: i for i, s in enumerate(order)}

    cur = random.choice(valid)
    seq = [random.choice(mapping[cur])]
    cr = rank[cur]

    for _ in range(1, count):
        allowed = [s for s in valid if (rank[s] >= cr if pattern=="aaroh" else rank[s] <= cr)]
        if not allowed:
            allowed = valid
        cur = random.choice(allowed)
        seq.append(random.choice(mapping[cur]))
        cr = rank[cur]
    return seq

# ==========================================
# Streamlit UI
# ==========================================

st.title("🎵 Sargam Trainer — Debesh's Version")

st.markdown("Practice random sargam note recognition with tempo control.")

# ----------- Settings Sidebar -----------
st.sidebar.header("Settings")

mode = st.sidebar.selectbox("Mode", ["free", "raga"])

sa = st.sidebar.selectbox("Sa (Base Note)", list(WESTERN_SA_MAP.keys()))

if mode == "free":
    komal = st.sidebar.checkbox("Include komal/teevra notes", True)
else:
    raga = st.sidebar.selectbox("Select Raga", list(RAGA_DEF.keys()))
    pattern = st.sidebar.selectbox("Pattern", ["free", "aaroh", "avroh"])

octaves = st.sidebar.multiselect("Octaves", ["L","M","U"], default=["M"])

count = st.sidebar.slider("Notes per sequence", 1, 20, 8)

bpm = st.sidebar.slider("Tempo (BPM)", 40, 200, 80)
beats_per_note = st.sidebar.slider("Beats per Note", 0.25, 4.0, 1.0)

duration = 60 / bpm * beats_per_note
st.sidebar.write(f"Note Duration: {duration:.2f} sec")

# ----------- Generate Sequence -----------
if st.button("Generate New Sequence"):
    sa_freq = WESTERN_SA_MAP[sa]

    if mode == "free":
        pool = build_pool_free(sa_freq, komal, octaves)
        seq = generate_free(pool, count)
    else:
        pool = build_pool_raga(sa_freq, raga, octaves)
        seq = generate_raga(pool, count, raga, pattern)

    st.session_state["seq"] = seq
    st.session_state["pool"] = pool
    st.success("New sequence generated!")


# ----------- Playback -----------


if "seq" in st.session_state:
    seq = st.session_state["seq"]

    if st.button("▶ Play Sequence"):
        pool = st.session_state.get("pool")
        if pool is None:
            st.error("Please generate a sequence first.")
        else:
            for s in seq:
                play_tone(pool[s], duration)

    if st.button("↺ Replay"):
        pool = st.session_state.get("pool")
        if pool is None:
            st.error("Please generate a sequence first.")
        else:
            for s in seq:
                play_tone(pool[s], duration)

    if st.button("👁 Reveal Notes"):
        st.write("Sequence:", " ".join(seq))
