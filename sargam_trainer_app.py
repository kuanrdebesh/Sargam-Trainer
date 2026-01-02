import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import io
import wave
import random
import base64

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

# ==========================================
# Audio (Streamlit Cloud friendly)
# - Generates WAV bytes and plays in browser via st.audio()
# ==========================================

def _tone_wave(freq: float, duration: float, sr: int = SAMPLE_RATE, volume: float = VOLUME) -> np.ndarray:
    """Return a mono waveform in float32 range [-1, 1]."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave_ = np.sin(2 * np.pi * freq * t) * volume
    return wave_.astype(np.float32)

def _wav_bytes(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Convert float32 waveform [-1,1] to 16-bit PCM WAV bytes."""
    if waveform.size == 0:
        return b""
    pcm = np.clip(waveform, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()

def build_sequence_wav_bytes(freqs: list[float], note_duration: float, gap: float = 0.03) -> bytes:
    """Build one continuous WAV (sequence), so UI shows a single audio player."""
    parts: list[np.ndarray] = []
    if gap and gap > 0:
        silence = np.zeros(int(SAMPLE_RATE * gap), dtype=np.float32)
    else:
        silence = None

    for f in freqs:
        parts.append(_tone_wave(f, note_duration))
        if silence is not None:
            parts.append(silence)

    waveform = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    return _wav_bytes(waveform, SAMPLE_RATE)


def render_audio_with_highlight(
    wav_bytes: bytes,
    labels: list[str],
    note_duration: float,
    gap: float,
    loop: bool = False,
    element_id: str = "player",
):
    """Render a single HTML audio player and highlight the current note as it plays."""
    if not wav_bytes:
        st.warning("No audio to play.")
        return

    b64 = base64.b64encode(wav_bytes).decode("ascii")
    total_step = max(note_duration + gap, 0.0001)

    # Simple inline HTML + JS. Works on Streamlit Cloud.
    html = f"""
    <div style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;">
      <audio id="{element_id}" controls {"loop" if loop else ""} style="width: 100%;">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
      </audio>

      <div id="{element_id}-notes" style="margin-top: 10px; line-height: 2.2;">
        {''.join([f'<span class="note-chip" data-idx="{i}" style="display:inline-block;padding:4px 10px;margin:4px;border-radius:999px;border:1px solid #ddd;">{lab}</span>' for i, lab in enumerate(labels)])}
      </div>
    </div>

    <script>
      (function() {{
        const audio = document.getElementById("{element_id}");
        const chips = Array.from(document.querySelectorAll("#{element_id}-notes .note-chip"));
        const step = {total_step};

        function clearActive() {{
          chips.forEach(c => {{
            c.style.background = "";
            c.style.borderColor = "#ddd";
            c.style.fontWeight = "400";
          }});
        }}

        function setActive(i) {{
          clearActive();
          if (i >= 0 && i < chips.length) {{
            const c = chips[i];
            c.style.background = "rgba(255, 210, 0, 0.25)";
            c.style.borderColor = "rgba(255, 160, 0, 0.8)";
            c.style.fontWeight = "600";
          }}
        }}

        let lastIdx = -1;

        function tick() {{
          if (!audio || audio.paused) return;
          const t = audio.currentTime || 0;
          const idx = Math.floor(t / step);
          if (idx !== lastIdx) {{
            lastIdx = idx;
            setActive(idx);
          }}
        }}

        audio.addEventListener("play", () => {{
          lastIdx = -1;
          tick();
        }});

        audio.addEventListener("timeupdate", tick);

        audio.addEventListener("ended", () => {{
          clearActive();
        }});

        // click a note to jump
        chips.forEach(c => {{
          c.addEventListener("click", () => {{
            const i = parseInt(c.getAttribute("data-idx"));
            audio.currentTime = i * step;
            audio.play();
          }});
        }});
      }})();
    </script>
    """

    components.html(html, height=150 + (len(labels)//6)*30)






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



# ----------- Playback Preferences -----------
st.sidebar.subheader("Playback")
play_mode = st.sidebar.radio(
    "Playback mode",
    ["All at once", "Each note individually"],
    index=0,
    help="All at once plays one continuous audio clip. Individual mode shows one player per note.",
)

st.sidebar.subheader("Tanpura (Drone)")
tanpura_on = st.sidebar.checkbox("Enable Tanpura", value=False)
tanpura_level = st.sidebar.slider("Tanpura volume", 0.0, 1.0, 0.25, 0.05) if tanpura_on else 0.0

def build_tanpura_wav_bytes(sa_freq: float, seconds: float = 60.0, sr: int = SAMPLE_RATE, level: float = 0.25) -> bytes:
    """A simple tanpura-style drone (Sa + Pa + upper Sa)."""
    if level <= 0:
        return b""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    # Sa, Pa, upper Sa
    pa = sa_freq * (3/2)
    upper_sa = sa_freq * 2
    drone = (
        np.sin(2 * np.pi * sa_freq * t) +
        0.7 * np.sin(2 * np.pi * pa * t) +
        0.5 * np.sin(2 * np.pi * upper_sa * t)
    )
    # gentle amplitude modulation to feel less static
    mod = 0.85 + 0.15 * np.sin(2 * np.pi * 0.25 * t)
    drone = drone * mod
    drone = drone / np.max(np.abs(drone) + 1e-9)
    drone = drone.astype(np.float32) * (VOLUME * level)
    return _wav_bytes(drone, sr=sr)

# ----------- Playback Output (placeholders to avoid stacking) -----------
audio_area = st.empty()
tuning_area = st.empty()
tanpura_area = st.empty()

# Render tanpura if enabled
if tanpura_on:
    tanp_wav = build_tanpura_wav_bytes(WESTERN_SA_MAP[sa], seconds=120.0, level=tanpura_level)
    if tanp_wav:
        with tanpura_area.container():
            st.markdown("**Tanpura (loop)**")
            # Use the HTML player so we can loop
            render_audio_with_highlight(
                tanp_wav,
                labels=["Sa (drone)"],
                note_duration=120.0,
                gap=0.0,
                loop=True,
                element_id="tanpura",
            )

# ----------- Sequence Playback -----------
if "seq" in st.session_state:
    seq = st.session_state["seq"]
    pool = st.session_state.get("pool")

    if st.button("▶ Play Sequence"):
        if pool is None:
            st.error("Please generate a sequence first.")
        else:
            with audio_area.container():
                if play_mode == "Each note individually":
                    st.markdown("**Sequence (individual notes)**")
                    for i, sym in enumerate(seq, start=1):
                        st.write(f"{i}. {sym}")
                        wav = build_sequence_wav_bytes([pool[sym]], note_duration=duration)
                        st.audio(wav, format="audio/wav")
                else:
                    st.markdown("**Sequence (all at once)**")
                    wav = build_sequence_wav_bytes([pool[s] for s in seq], note_duration=duration)
                    render_audio_with_highlight(
                        wav_bytes=wav,
                        labels=seq,
                        note_duration=duration,
                        gap=0.03,
                        loop=False,
                        element_id="sequence",
                    )

# ----------- Ear Tuning (Play all notes available) -----------
st.markdown("---")
st.subheader("🎧 Ear Tuning")

st.caption("Play all available notes for the selected settings to tune your ear. The currently playing note will highlight.")

if st.button("Play all available notes"):
    pool = st.session_state.get("pool")

    # If user hasn't generated a sequence yet, build a pool from current settings.
    if pool is None:
        sa_freq = WESTERN_SA_MAP[sa]
        if mode == "free":
            pool = build_pool_free(sa_freq, komal, octaves)
        else:
            pool = build_pool_raga(sa_freq, raga, octaves)

    # sort by frequency so it's musically sensible
    items = sorted(pool.items(), key=lambda kv: kv[1])
    labels = [k for k,_ in items]
    freqs = [v for _,v in items]

    wav = build_sequence_wav_bytes(freqs, note_duration=duration)
    with tuning_area.container():
        render_audio_with_highlight(
            wav_bytes=wav,
            labels=labels,
            note_duration=duration,
            gap=0.03,
            loop=False,
            element_id="tuning",
        )

if st.button("👁 Reveal Notes"):
    if "seq" in st.session_state:
        st.write("Sequence:", " ".join(st.session_state["seq"]))
    else:
        st.info("Generate a sequence first.")

