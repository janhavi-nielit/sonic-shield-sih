import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr
import io
import os
import time

st.set_page_config(page_title="Team SonicShield - AI ANC", layout="wide")
st.title("🛡️ Tactical Audio Armor: Live AI Noise Suppression Engine")
st.markdown("### Developed by Team SonicShield for Smart India Hackathon")
st.write("---")

sr = 16000

# ---------- Core processing functions ----------

def suppress_noise(signal, strength=0.8, stationary=False):
    """Real noise suppression using spectral gating (noisereduce library)."""
    return nr.reduce_noise(
        y=signal,
        sr=sr,
        stationary=stationary,
        prop_decrease=strength,
        n_fft=512,
        win_length=512,
        hop_length=128,
    )

def measure_latency(func, *args, **kwargs):
    """Times how long the suppression function actually takes to run."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms

def compute_snr(clean, test):
    """Real SNR — requires a ground-truth clean reference (used for presets only)."""
    n = min(len(clean), len(test))
    clean, test = clean[:n], test[:n]
    noise = test - clean
    return 10 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-10))

def compute_snr_proxy(signal):
    """No ground truth available for live mic input — proxy metric only.
    Ratio of energy in the speech band (300-3400 Hz) vs full band."""
    stft = np.abs(librosa.stft(signal, n_fft=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    speech_band = (freqs >= 300) & (freqs <= 3400)
    speech_energy = np.sum(stft[speech_band] ** 2)
    total_energy = np.sum(stft ** 2) + 1e-10
    ratio = speech_energy / total_energy
    return 10 * np.log10(ratio / (1 - ratio + 1e-10) + 1e-10)

def plot_wave(signal, color):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(signal, color=color, linewidth=0.5)
    ax.axis('off')
    return fig

def plot_spectrogram(signal, title, cmap="magma"):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=512, hop_length=128)), ref=np.max)
    librosa.display.specshow(D, sr=sr, hop_length=128, x_axis='time', y_axis='hz', ax=ax, cmap=cmap)
    ax.set_title(title, fontsize=9, color='white')
    fig.patch.set_alpha(0)
    return fig

# ---------- UI ----------

tab1, tab2 = st.tabs(["📁 Preset Scenarios", "🎙️ Live Mic Input"])

# ===== TAB 1: Preset noisy scenarios with known clean reference =====
with tab1:
    st.sidebar.header("🕹️ Tactical Control Panel")
    noise_scenario = st.sidebar.selectbox(
        "Select Tactical Scenario", ["Military Drone Noise", "Artillery Ambient"]
    )
    alpha = st.sidebar.slider("Suppression Strength", 0.3, 0.95, 0.8, 0.05, key="alpha_preset")
    stationary_preset = st.sidebar.checkbox(
        "Treat noise as stationary (steady hum)", value=False, key="stat_preset"
    )

    noisy_path = "noisybackground.wav" if noise_scenario == "Military Drone Noise" else "noisybackground1.wav"
    clean_path = "speech.wav"

    if os.path.exists(noisy_path) and os.path.exists(clean_path):
        noisy, _ = librosa.load(noisy_path, sr=sr)
        clean, _ = librosa.load(clean_path, sr=sr)

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🎯 Input: Corrupted Tactical Communication ({noise_scenario})")
            st.audio(noisy_path)
            st.pyplot(plot_wave(noisy, "crimson"))
            st.pyplot(plot_spectrogram(noisy, "Input Spectrogram", cmap="Reds"))

        if st.button("🚀 Run AI Noise Suppression Filter", type="primary", key="run_preset"):
            with st.spinner("Analyzing Speech Domains & Removing Noise..."):
                enhanced, latency_ms = measure_latency(
                    suppress_noise, noisy, strength=alpha, stationary=stationary_preset
                )
                enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-9) * 0.9
                sf.write("enhanced_output.wav", enhanced, sr)

                audio_duration_ms = (len(noisy) / sr) * 1000
                snr_before = compute_snr(clean, noisy)
                snr_after = compute_snr(clean, enhanced)
                diff = np.mean(np.abs(noisy[:len(enhanced)] - enhanced))

            with col2:
                st.success("✨ Output: Isolated Clean Speech Signal")
                st.audio("enhanced_output.wav")
                st.pyplot(plot_wave(enhanced, "green"))
                st.pyplot(plot_spectrogram(enhanced, "Output Spectrogram", cmap="Greens"))

            st.write("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📊 SNR Before", f"{snr_before:.1f} dB")
            m2.metric("📊 SNR After", f"{snr_after:.1f} dB", delta=f"{snr_after - snr_before:+.1f} dB")
            m3.metric("⚡ Suppression Strength", f"{alpha:.2f}")
            m4.metric("⏱️ Processing Time", f"{latency_ms:.0f} ms",
                       delta=f"for {audio_duration_ms:.0f} ms audio", delta_color="off")

            if diff < 0.001:
                st.warning("⚠️ Very small change detected — try increasing Suppression Strength.")
    else:
        st.error("Audio files not found in repo (noisybackground.wav / speech.wav).")

# ===== TAB 2: Live mic input, button-gated processing =====
with tab2:
    st.subheader("🎙️ Speak Live — Real Noise Suppression on Your Voice")
    alpha_live = st.slider("Suppression Strength", 0.3, 0.95, 0.8, 0.05, key="alpha_live")
    stationary_live = st.sidebar.checkbox(
        "Treat noise as stationary (steady hum)", value=False, key="stat_live"
    )
    audio_value = st.audio_input("Record live audio (speak with background noise for best demo)")

    if audio_value is not None:
        y, _ = librosa.load(io.BytesIO(audio_value.getvalue()), sr=sr)

        col1, col2 = st.columns(2)
        with col1:
            st.info("🎯 Live Captured Input")
            st.audio(audio_value)
            st.pyplot(plot_wave(y, "crimson"))
            st.pyplot(plot_spectrogram(y, "Input Spectrogram", cmap="Reds"))

        if st.button("🚀 Run AI Noise Suppression Filter", type="primary", key="run_live"):
            with st.spinner("Analyzing Speech Domains & Removing Noise..."):
                enhanced, latency_ms = measure_latency(
                    suppress_noise, y, strength=alpha_live, stationary=stationary_live
                )
                enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-9) * 0.9
                sf.write("live_enhanced.wav", enhanced, sr)

                audio_duration_ms = (len(y) / sr) * 1000
                proxy_before = compute_snr_proxy(y)
                proxy_after = compute_snr_proxy(enhanced)
                diff = np.mean(np.abs(y[:len(enhanced)] - enhanced))

            with col2:
                st.success("✨ Suppressed Output")
                st.audio("live_enhanced.wav")
                st.pyplot(plot_wave(enhanced, "green"))
                st.pyplot(plot_spectrogram(enhanced, "Output Spectrogram", cmap="Greens"))

            st.write("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📊 Speech-Band Clarity (before)", f"{proxy_before:.1f} dB")
            m2.metric("📊 Speech-Band Clarity (after)", f"{proxy_after:.1f} dB", delta=f"{proxy_after - proxy_before:+.1f} dB")
            m3.metric("⚡ Suppression Strength", f"{alpha_live:.2f}")
            m4.metric("⏱️ Processing Time", f"{latency_ms:.0f} ms",
                       delta=f"for {audio_duration_ms:.0f} ms audio", delta_color="off")

            if diff < 0.001:
                st.warning("⚠️ Very small change detected — try increasing Suppression Strength or check noisereduce install.")
