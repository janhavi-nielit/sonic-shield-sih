import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import io
import os

st.set_page_config(page_title="Team SonicShield - AI ANC", layout="wide")
st.title("🛡️ Tactical Audio Armor: Live AI Noise Suppression Engine")
st.markdown("### Developed by Team SonicShield for Smart India Hackathon")
st.write("---")

sr = 16000

def spectral_subtraction(noisy, alpha=2.0, noise_frames=10, n_fft=512, hop_length=128):
    stft = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_mag = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    enhanced_mag = np.maximum(mag - alpha * noise_mag, 0.05 * mag)
    enhanced_stft = enhanced_mag * np.exp(1j * phase)
    return librosa.istft(enhanced_stft, hop_length=hop_length, length=len(noisy))

def compute_snr_proxy(signal):
    # No clean reference available for live mic input, so use a proxy:
    # ratio of energy in speech band vs full band, as a rough intelligibility indicator
    stft = np.abs(librosa.stft(signal, n_fft=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    speech_band = (freqs >= 300) & (freqs <= 3400)
    speech_energy = np.sum(stft[speech_band] ** 2)
    total_energy = np.sum(stft ** 2) + 1e-10
    ratio = speech_energy / total_energy
    return 10 * np.log10(ratio / (1 - ratio + 1e-10) + 1e-10)

tab1, tab2 = st.tabs(["📁 Preset Scenarios", "🎙️ Live Mic Input"])

with tab1:
    st.sidebar.header("🕹️ Tactical Control Panel")
    noise_scenario = st.sidebar.selectbox("Select Tactical Scenario", ["Military Drone Noise", "Artillery Ambient"])
    alpha = st.sidebar.slider("Suppression Strength (α)", 1.0, 4.0, 2.0, 0.1, key="alpha_preset")

    noisy_path = "noisybackground.wav" if noise_scenario == "Military Drone Noise" else "noisybackground1.wav"
    clean_path = "speech.wav"

    if os.path.exists(noisy_path) and os.path.exists(clean_path):
        noisy, _ = librosa.load(noisy_path, sr=sr)
        clean, _ = librosa.load(clean_path, sr=sr)
        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🎯 Input: {noise_scenario}")
            st.audio(noisy_path)
            fig, ax = plt.subplots(figsize=(5, 2)); ax.plot(noisy, color="crimson", linewidth=0.5); ax.axis('off')
            st.pyplot(fig)

        if st.button("🚀 Run AI Noise Suppression Filter", type="primary", key="run_preset"):
            enhanced = spectral_subtraction(noisy, alpha=alpha)
            enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-9) * 0.9
            sf.write("enhanced_output.wav", enhanced, sr)
            snr_before = 10 * np.log10(np.sum(clean**2) / (np.sum((noisy-clean)**2) + 1e-10))
            snr_after = 10 * np.log10(np.sum(clean**2) / (np.sum((enhanced[:n]-clean)**2) + 1e-10))
            with col2:
                st.success("✨ Output: Isolated Clean Speech Signal")
                st.audio("enhanced_output.wav")
                fig2, ax2 = plt.subplots(figsize=(5, 2)); ax2.plot(enhanced, color="green", linewidth=0.5); ax2.axis('off')
                st.pyplot(fig2)
            m1, m2, m3 = st.columns(3)
            m1.metric("📊 SNR Before", f"{snr_before:.1f} dB")
            m2.metric("📊 SNR After", f"{snr_after:.1f} dB", delta=f"{snr_after-snr_before:+.1f} dB")
            m3.metric("⚡ Suppression Factor (α)", f"{alpha:.1f}")

with tab2:
    st.subheader("🎙️ Speak Live — Real Noise Suppression on Your Voice")
    alpha_live = st.slider("Suppression Strength (α)", 1.0, 4.0, 2.0, 0.1, key="alpha_live")
    audio_value = st.audio_input("Record live audio (speak with background noise for best demo)")

    if audio_value is not None:
        y, _ = librosa.load(io.BytesIO(audio_value.getvalue()), sr=sr)

        col1, col2 = st.columns(2)
        with col1:
            st.info("🎯 Live Captured Input")
            st.audio(audio_value)
            fig, ax = plt.subplots(figsize=(5, 2)); ax.plot(y, color="crimson", linewidth=0.5); ax.axis('off')
            st.pyplot(fig)

        enhanced = spectral_subtraction(y, alpha=alpha_live)
        enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-9) * 0.9
        sf.write("live_enhanced.wav", enhanced, sr)

        with col2:
            st.success("✨ Suppressed Output")
            st.audio("live_enhanced.wav")
            fig2, ax2 = plt.subplots(figsize=(5, 2)); ax2.plot(enhanced, color="green", linewidth=0.5); ax2.axis('off')
            st.pyplot(fig2)

        proxy_before = compute_snr_proxy(y)
        proxy_after = compute_snr_proxy(enhanced)
        m1, m2, m3 = st.columns(3)
        m1.metric("📊 Speech-Band Clarity (before)", f"{proxy_before:.1f} dB")
        m2.metric("📊 Speech-Band Clarity (after)", f"{proxy_after:.1f} dB", delta=f"{proxy_after-proxy_before:+.1f} dB")
        m3.metric("⚡ Suppression Factor (α)", f"{alpha_live:.1f}")
