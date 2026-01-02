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
NOTE_VOLUME = 0.25
TANPURA_VOLUME = 0.9
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

def _adsr_envelope(n: int, sr: int, attack: float = 0.01, decay: float = 0.08, sustain: float = 0.70, release: float = 0.14) -> np.ndarray:
    """Simple ADSR envelope."""
    attack_n = max(1, int(sr * attack))
    decay_n = max(1, int(sr * decay))
    release_n = max(1, int(sr * release))
    sustain_n = max(0, n - (attack_n + decay_n + release_n))

    a = np.linspace(0.0, 1.0, attack_n, endpoint=False)
    d = np.linspace(1.0, sustain, decay_n, endpoint=False)
    s = np.full(sustain_n, sustain, dtype=np.float32)
    r = np.linspace(sustain, 0.0, release_n, endpoint=True)

    env = np.concatenate([a, d, s, r]).astype(np.float32)
    if env.size < n:
        env = np.pad(env, (0, n - env.size))
    else:
        env = env[:n]
    return env

def _one_pole_lowpass(x: np.ndarray, sr: int, cutoff_hz: float = 3800.0) -> np.ndarray:
    """Light low-pass filter for a softer, less 'beepy' tone."""
    if cutoff_hz <= 0:
        return x
    # One-pole low-pass: y[n] = (1-a)*x[n] + a*y[n-1]
    a = float(np.exp(-2.0 * np.pi * cutoff_hz / sr))
    y = np.empty_like(x, dtype=np.float32)
    y0 = 0.0
    one_minus_a = 1.0 - a
    for i in range(x.size):
        y0 = one_minus_a * x[i] + a * y0
        y[i] = y0
    return y


def render_looping_audio(
    wav_bytes: bytes,
    volume: float = 0.6,
    element_id: str = "looping-audio",
    label: str | None = None,
):
    """Render a looping audio player that can play in parallel with other players.

    Why HTML instead of st.audio?
    - We can set loop=true reliably without generating a very long file.
    - We can set the element volume via JS without regenerating audio.
    """
    if not wav_bytes:
        st.warning("No audio to play.")
        return

    vol = float(np.clip(volume, 0.0, 1.0))
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    title = f"<div style='font-weight:600;margin-bottom:6px'>{label}</div>" if label else ""

    html = f"""
    <div style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;">
      {title}
      <audio id="{element_id}" controls loop style="width: 100%;">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
      </audio>
    </div>
    <script>
      (function() {{
        const a = document.getElementById('{element_id}');
        if (a) a.volume = {vol};
      }})();
    </script>
    """
    components.html(html, height=90)

def _tiny_reverb(x: np.ndarray, sr: int) -> np.ndarray:
    """Very small reverb/room feel using a few short delays."""
    delays_ms = [28, 41, 57]
    gains = [0.25, 0.18, 0.12]
    y = x.astype(np.float32).copy()
    for d_ms, g in zip(delays_ms, gains):
        d = int(sr * (d_ms / 1000.0))
        if d <= 0 or d >= y.size:
            continue
        y[d:] += g * x[:-d]
    # gentle damping
    y = _one_pole_lowpass(y, sr, cutoff_hz=5200.0)
    return y

def _tone_wave(freq: float, duration: float, sr: int = SAMPLE_RATE, volume: float = NOTE_VOLUME) -> np.ndarray:
    """More natural tone: harmonics + slight detune + ADSR + soft filtering."""
    n = int(sr * duration)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    t = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)

    # Add harmonics (string-like): 1..8 with rolloff.
    # Slight random detune per note to avoid robotic feel.
    detune_cents = np.random.uniform(-4.0, 4.0)
    detune = 2 ** (detune_cents / 1200.0)
    f0 = freq * detune

    sig = np.zeros(n, dtype=np.float32)
    # Fundamental stronger, higher harmonics softer
    for k in range(1, 9):
        amp = 1.0 / (k ** 1.25)
        # tiny per-harmonic detune for richness
        fk = f0 * k * (1.0 + np.random.uniform(-0.0008, 0.0008))
        sig += amp * np.sin(2 * np.pi * fk * t)

    # Breath/noise very subtly (attack realism)
    noise = np.random.normal(0.0, 1.0, n).astype(np.float32)
    noise_env = _adsr_envelope(n, sr, attack=0.002, decay=0.05, sustain=0.0, release=0.05)
    sig += 0.03 * noise * noise_env

    # Envelope & tone shaping
    env = _adsr_envelope(n, sr, attack=0.008, decay=0.10, sustain=0.72, release=0.16)
    sig *= env

    # Softer top-end + a hint of room
    sig = _one_pole_lowpass(sig, sr, cutoff_hz=4200.0)
    sig = _tiny_reverb(sig, sr)

    # Normalize gently and apply volume
    peak = float(np.max(np.abs(sig))) if sig.size else 1.0
    if peak > 0:
        sig = sig / peak
    sig *= float(volume)
    return sig.astype(np.float32)

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
tanpura_level = st.sidebar.slider("Tanpura volume", 0.0, 1.0, 0.6, 0.05) if tanpura_on else 0.0

def _tanpura_pluck(freq: float, sr: int, pluck_len: float = 2.4) -> np.ndarray:
    """A tanpura-like pluck with long decay and rich harmonics."""
    n = int(sr * pluck_len)
    t = np.linspace(0, pluck_len, n, endpoint=False, dtype=np.float32)

    # Harmonics for a string drone (tanpura-ish)
    sig = np.zeros(n, dtype=np.float32)
    for k in range(1, 13):
        amp = 1.0 / (k ** 1.15)
        # slightly stronger odd harmonics for woody feel
        if k % 2 == 1:
            amp *= 1.12
        fk = freq * k * (1.0 + np.random.uniform(-0.0004, 0.0004))
        sig += amp * np.sin(2 * np.pi * fk * t)

    # Very long, smooth decay envelope
    env = _adsr_envelope(n, sr, attack=0.004, decay=0.18, sustain=0.55, release=1.2)
    sig *= env

    # Slight buzzing/noise at the beginning (jawari feel)
    noise = np.random.normal(0.0, 1.0, n).astype(np.float32)
    buzz_env = _adsr_envelope(n, sr, attack=0.001, decay=0.07, sustain=0.0, release=0.05)
    sig += 0.06 * noise * buzz_env

    sig = _one_pole_lowpass(sig, sr, cutoff_hz=3200.0)
    return sig.astype(np.float32)

def build_tanpura_wav_bytes(sa_freq: float, seconds: float = 60.0, sr: int = SAMPLE_RATE, level: float = 0.25) -> bytes:
    """Tanpura-style drone using repeated plucks (Sa–Pa–Sa–Sa')."""
    if level <= 0:
        return b""
    total_n = int(sr * seconds)
    if total_n <= 0:
        return b""

    out = np.zeros(total_n, dtype=np.float32)

    pa = sa_freq * (3/2)
    upper_sa = sa_freq * 2.0

    # Typical tanpura cycle (approx): Sa, Pa, Sa, upper Sa (varies by style)
    cycle = [sa_freq, pa, sa_freq, upper_sa]
    interval = 1.65  # seconds between plucks (feel free to tweak)
    pluck_len = 2.8  # pluck tail overlaps next pluck

    t0 = 0.0
    idx = 0
    while t0 < seconds:
        f = cycle[idx % len(cycle)]
        pl = _tanpura_pluck(f, sr=sr, pluck_len=pluck_len)
        start = int(t0 * sr)
        end = min(total_n, start + pl.size)
        if start < total_n:
            out[start:end] += pl[:end-start]
        t0 += interval
        idx += 1

    # Gentle slow amplitude motion
    t = np.linspace(0, seconds, total_n, endpoint=False, dtype=np.float32)
    mod = 0.92 + 0.08 * np.sin(2 * np.pi * 0.18 * t + 0.5)
    out *= mod

    # A bit more room + normalize
    out = _tiny_reverb(out, sr)
    peak = float(np.max(np.abs(out))) if out.size else 1.0
    if peak > 0:
        out = out / peak

    out = out.astype(np.float32) * (TANPURA_VOLUME * level)
    return _wav_bytes(out, sr=sr)


@st.cache_data(show_spinner=False)
def cached_tanpura_clip(sa_freq: float) -> bytes:
    """Small tanpura clip (looped in browser).

    Keeping it short avoids big CPU work and prevents the UI from becoming
    unresponsive. We loop it in the browser using an HTML audio element.
    """
    return build_tanpura_wav_bytes(sa_freq, seconds=12.0, level=1.0)


# ----------- Tanpura (plays in parallel) -----------
# Render in the main area using an HTML <audio loop> element.
# This keeps the UI responsive and lets Tanpura play alongside other players.
if tanpura_on and tanpura_level > 0:
    st.markdown("### 🎶 Tanpura")
    st.caption("Press play once — it will loop, and you can still play sequences/ear-tuning on top.")
    tanp_wav = cached_tanpura_clip(WESTERN_SA_MAP[sa])
    render_looping_audio(wav_bytes=tanp_wav, volume=tanpura_level, element_id="tanpura")

# We render Tanpura in the *main* area using an HTML audio element with loop=true.
# This prevents long audio generation that can freeze the app, and it plays in
# parallel with the other audio players.

# ----------- Sequence Playback -----------
audio_area = st.empty()
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
                        st.write(f"Note {i}")
                        wav = build_sequence_wav_bytes([pool[sym]], note_duration=duration)
                        st.audio(wav, format="audio/wav")
                else:
                    st.markdown("**Sequence (all at once)**")
                    st.caption("Notes are hidden until you click **Reveal Notes**.")
                    wav = build_sequence_wav_bytes([pool[s] for s in seq], note_duration=duration)
                    st.audio(wav, format="audio/wav")

# ----------- Ear Tuning (Play all notes available) -----------
st.markdown("---")
st.subheader("🎧 Ear Tuning")

tuning_area = st.empty()

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

