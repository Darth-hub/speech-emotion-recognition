"""
Streamlit demo for Speech Emotion Recognition.

Two input paths:
  1. Upload a WAV/MP3/OGG/FLAC file
  2. Record from the microphone (uses streamlit-mic-recorder)

For each prediction we display:
  - Top emotion + confidence
  - Bar chart of probabilities over all 7 classes
  - Mel spectrogram with the model's attention weights overlaid in red
    (this is our explainability story -- "the model focused here")

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make `src` importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import Predictor              # noqa: E402
from src.preprocessing import SAMPLE_RATE      # noqa: E402

CKPT_PATH = Path("models/checkpoints/best.pt")


@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor(CKPT_PATH)


def plot_attention(features: np.ndarray, attention: np.ndarray) -> plt.Figure:
    """Spectrogram on top, attention curve on bottom (shared time axis)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    # Show only the log-Mel block (first 64 rows of the 116-feature stack).
    ax1.imshow(features[:64], aspect="auto", origin="lower", cmap="magma")
    ax1.set_ylabel("Mel band"); ax1.set_title("Log-Mel spectrogram")

    t = np.linspace(0, features.shape[1], len(attention))
    ax2.plot(t, attention, color="crimson", linewidth=2)
    ax2.fill_between(t, attention, alpha=0.3, color="crimson")
    ax2.set_xlabel("Frame"); ax2.set_ylabel("Attention")
    ax2.set_title("Where the model looked")
    fig.tight_layout()
    return fig


def run_prediction(audio_bytes: bytes, suffix: str = ".wav") -> None:
    if not CKPT_PATH.exists():
        st.error(f"Checkpoint not found at {CKPT_PATH}. Train the model first "
                 "(`python -m src.train`).")
        return

    predictor = get_predictor()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name

    with st.spinner("Analyzing..."):
        result = predictor.predict(tmp_path)

    # Headline result.
    st.success(f"**Predicted emotion:** {result['label']}  "
               f"(confidence {result['probs'][result['label']]:.1%})")

    # Probability bar chart.
    st.subheader("Probabilities")
    probs_df = (pd.DataFrame.from_dict(result["probs"], orient="index", columns=["probability"])
                  .sort_values("probability", ascending=False))
    st.bar_chart(probs_df)

    # Spectrogram + attention.
    st.subheader("Explainability: attention over time")
    st.pyplot(plot_attention(result["features"], result["attention"]))


def main() -> None:
    st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
    st.title("🎙️ Speech Emotion Recognition")
    st.write("Upload an audio clip or record from your microphone. The model returns "
             "the predicted emotion along with an attention map showing which moments "
             "drove the prediction.")

    tab_upload, tab_record = st.tabs(["📁 Upload", "🎤 Record"])

    with tab_upload:
        f = st.file_uploader("Audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])
        if f is not None:
            st.audio(f)
            run_prediction(f.read(), suffix=Path(f.name).suffix or ".wav")

    with tab_record:
        try:
            from streamlit_mic_recorder import mic_recorder
        except ImportError:
            st.info("Install `streamlit-mic-recorder` for microphone recording: "
                    "`pip install streamlit-mic-recorder`")
            return
        rec = mic_recorder(start_prompt="🔴 Start", stop_prompt="⏹ Stop",
                           just_once=True, format="wav")
        if rec and rec.get("bytes"):
            st.audio(io.BytesIO(rec["bytes"]))
            run_prediction(rec["bytes"], suffix=".wav")


if __name__ == "__main__":
    main()
