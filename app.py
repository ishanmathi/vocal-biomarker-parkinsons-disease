import streamlit as st
import librosa
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Vocal Biomarker – Parkinson's Detection")
st.title("🎤 Vocal Biomarker – Parkinson's Detection")
st.write("Upload a voice sample")

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "ogg", "m4a"]
)

def extract_features(y, sr):
    jitter = np.mean(np.abs(np.diff(y)))
    shimmer = np.std(y)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rmse = np.mean(librosa.feature.rms(y=y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))

    return jitter, shimmer, zcr, rmse, mfcc

def detect_parkinsons(jitter, shimmer, zcr):
    score = 0
    if jitter > 0.03:
        score += 1
    if shimmer > 0.05:
        score += 1
    if zcr < 0.05:
        score += 1

    if score >= 2:
        return "⚠️ Parkinson’s Indicators Detected", "High Risk"
    else:
        return "✅ Voice Appears Normal", "Low Risk"

if uploaded_file is not None:
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

        # Load audio directly using librosa
        y, sr = librosa.load(audio_path, sr=16000)

        st.audio(uploaded_file)
        st.success("Audio loaded successfully 🎧")

        # Feature extraction
        jitter, shimmer, zcr, rmse, mfcc = extract_features(y, sr)

        # Detection
        result, risk = detect_parkinsons(jitter, shimmer, zcr)

        st.subheader("🧠 Detection Result")
        st.markdown(f"### {result}")
        st.write(f"**Risk Level:** {risk}")

        with st.expander("🔍 Extracted Vocal Features"):
            st.write(f"Jitter: {jitter:.5f}")
            st.write(f"Shimmer: {shimmer:.5f}")
            st.write(f"Zero Crossing Rate: {zcr:.5f}")
            st.write(f"RMSE: {rmse:.5f}")
            st.write(f"MFCC Mean: {mfcc:.5f}")

        os.remove(audio_path)

    except Exception as e:
        st.error("❌ Error processing audio")
        st.exception(e)
