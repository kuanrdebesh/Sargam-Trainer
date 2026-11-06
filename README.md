# Sargam Trainer

Raga practice without a tanpura app, a teacher, or a fixed key. Pick your Sa, pick a raga, and drill the swaras — aroha, avaroha, and ear-tuning exercises — with tones generated on the fly.

Built because every practice tool I tried assumed a Western scale, a fixed tonic, or a library of recorded samples. Indian classical practice needs none of those. It needs the right twelve intervals relative to wherever your Sa sits today.

### What it does

- **Any Sa.** Choose your tonic from the twelve Western notes; every swara is derived from it.
- **Eight ragas.** Yaman, Bhairavi, Bhoop, Bhimpalasi, Durga, Khamaj, Puriya Dhanashree, Bageshree — each with its own aroha and avaroha.
- **Three octaves.** Mandra, madhya and taar, notated with the usual comma and apostrophe.
- **Ear tuning.** Hear a swara, name it, check yourself.

### How it's built

A single Streamlit app. Notes are synthesised as sine waves at the correct frequency ratios and written to WAV in memory — no sample library, so any Sa and any tempo works without new assets.

### Run it

```bash
pip install -r requirements.txt
streamlit run sargam_trainer_app.py
```

Needs Python 3.9+. Nothing else.

### Status

Working and used — by me, for my own practice. Not packaged for anyone else yet.
