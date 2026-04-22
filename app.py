import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import joblib
import time
from src.feature_extraction import extract_features_from_audio

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aura · Emotion Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/emotion_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

emotion_map = {
    "01": ("Neutral", "😐", "#94a3b8", "A state of emotional equilibrium"),
    "02": ("Calm", "😌", "#67e8f9", "Relaxed and at peace"),
    "03": ("Happy", "😄", "#fbbf24", "Radiating positive energy"),
    "04": ("Sad", "😢", "#818cf8", "A contemplative melancholy"),
    "05": ("Angry", "😠", "#f87171", "Intense emotional arousal"),
    "06": ("Fearful", "😨", "#c084fc", "Heightened state of alertness"),
    "07": ("Disgust", "🤢", "#4ade80", "Aversion response detected"),
    "08": ("Surprised", "😲", "#fb923c", "Unexpected stimulus detected"),
}

# ── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #111119;
        --bg-card: rgba(255, 255, 255, 0.03);
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-glow: rgba(255, 255, 255, 0.12);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        --accent-blue: #667eea;
        --accent-purple: #764ba2;
        --accent-pink: #f093fb;
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
        --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    /* ── Global Overrides ── */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .stApp > header { background: transparent !important; }

    .block-container {
        max-width: 1100px !important;
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer, .stDeployButton { display: none !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.3);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(102, 126, 234, 0.5); }

    /* ── Hero Section ── */
    .hero-section {
        text-align: center;
        padding: 3rem 1rem 2rem;
        position: relative;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 18px;
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 100px;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent-blue);
        margin-bottom: 1.75rem;
        animation: fadeInDown 0.8s ease-out;
    }

    .hero-badge::before {
        content: '';
        width: 6px;
        height: 6px;
        background: var(--accent-blue);
        border-radius: 50%;
        animation: pulse-dot 2s ease-in-out infinite;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.5); }
    }

    .hero-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(2.5rem, 6vw, 4.2rem);
        font-weight: 600;
        line-height: 1.1;
        color: var(--text-primary);
        margin-bottom: 1.25rem;
        animation: fadeInUp 0.8s ease-out 0.15s both;
    }

    .hero-title .gradient-text {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 300;
        color: var(--text-secondary);
        max-width: 560px;
        margin: 0 auto 2.5rem;
        line-height: 1.7;
        animation: fadeInUp 0.8s ease-out 0.3s both;
    }

    /* ── Glassmorphism CTA Button ── */
    .glass-cta-wrapper {
        display: flex;
        justify-content: center;
        animation: fadeInUp 0.8s ease-out 0.45s both;
    }

    .glass-cta {
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 12px;
        padding: 18px 42px;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        color: #fff;
        cursor: pointer;
        border: none;
        outline: none;
        text-decoration: none;

        /* ── Glassmorphism ── */
        background: rgba(102, 126, 234, 0.15);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 16px;
        box-shadow:
            0 8px 32px rgba(102, 126, 234, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 0 rgba(102, 126, 234, 0);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        overflow: hidden;
    }

    .glass-cta::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.08),
            transparent
        );
        transition: left 0.6s ease;
    }

    .glass-cta:hover {
        background: rgba(102, 126, 234, 0.25);
        border-color: rgba(255, 255, 255, 0.3);
        box-shadow:
            0 12px 40px rgba(102, 126, 234, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.15),
            0 0 60px rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
    }

    .glass-cta:hover::before {
        left: 100%;
    }

    .glass-cta:active {
        transform: translateY(0);
        box-shadow:
            0 4px 16px rgba(102, 126, 234, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    .cta-icon {
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .cta-icon svg {
        width: 20px;
        height: 20px;
        fill: none;
        stroke: currentColor;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
    }

    /* ── Stats Bar ── */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 3rem;
        padding: 2rem 0;
        margin: 1rem 0 2.5rem;
        border-top: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        animation: fadeIn 1s ease-out 0.6s both;
    }

    .stat-item {
        text-align: center;
    }

    .stat-value {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 4px;
    }

    .stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
    }

    /* ── Recording Card ── */
    .recording-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem auto;
        max-width: 680px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.3s ease;
    }

    .recording-card:hover {
        border-color: var(--border-glow);
    }

    .recording-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.5rem;
    }

    .card-icon {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: rgba(102, 126, 234, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }

    .card-title {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .card-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: var(--text-muted);
    }

    /* ── Waveform Animation ── */
    .waveform-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 3px;
        padding: 2rem 0;
        min-height: 80px;
    }

    .wave-bar {
        width: 3px;
        border-radius: 3px;
        background: var(--accent-gradient);
        animation: wave-idle 1.5s ease-in-out infinite;
    }

    .wave-bar.active {
        animation: wave-active 0.4s ease-in-out infinite alternate;
    }

    @keyframes wave-idle {
        0%, 100% { height: 8px; opacity: 0.3; }
        50% { height: 16px; opacity: 0.5; }
    }

    @keyframes wave-active {
        0% { height: 8px; }
        100% { height: 48px; }
    }

    /* ── Result Card ── */
    .result-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        margin: 2rem auto;
        max-width: 680px;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: scaleIn 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--accent-gradient);
    }

    .result-emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: bounceIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55) 0.2s both;
    }

    .result-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }

    .result-emotion {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        animation: fadeInUp 0.5s ease-out 0.3s both;
    }

    .result-description {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 300;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        animation: fadeInUp 0.5s ease-out 0.4s both;
    }

    .confidence-bar-bg {
        width: 100%;
        max-width: 320px;
        height: 4px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 2px;
        margin: 0 auto;
        overflow: hidden;
    }

    .confidence-bar-fill {
        height: 100%;
        border-radius: 2px;
        background: var(--accent-gradient);
        animation: fillBar 1s ease-out 0.5s both;
    }

    @keyframes fillBar {
        from { width: 0; }
    }

    /* ── Emotion Grid ── */
    .emotion-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 2rem auto;
        max-width: 680px;
    }

    @media (max-width: 640px) {
        .emotion-grid { grid-template-columns: repeat(2, 1fr); }
    }

    .emotion-chip {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 12px 16px;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    .emotion-chip:hover {
        border-color: var(--border-glow);
        background: rgba(255, 255, 255, 0.05);
        transform: translateY(-1px);
    }

    .emotion-chip.active {
        border-color: rgba(102, 126, 234, 0.4);
        background: rgba(102, 126, 234, 0.08);
    }

    .chip-emoji { font-size: 1.2rem; }

    .chip-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-secondary);
    }

    /* ── Footer ── */
    .premium-footer {
        text-align: center;
        padding: 3rem 1rem 1rem;
        margin-top: 3rem;
    }

    .footer-brand {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }

    .footer-note {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        color: var(--text-muted);
        opacity: 0.5;
        letter-spacing: 0.05em;
    }

    /* ── Animations ── */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }

    @keyframes bounceIn {
        from { opacity: 0; transform: scale(0.3); }
        to { opacity: 1; transform: scale(1); }
    }

    /* ── Streamlit Button Override ── */
    .stButton > button {
        background: rgba(102, 126, 234, 0.15) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 16px !important;
        color: #fff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        padding: 16px 40px !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        width: auto !important;
        margin: 0 auto !important;
        display: block !important;
    }

    .stButton > button:hover {
        background: rgba(102, 126, 234, 0.25) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.35),
                    inset 0 1px 0 rgba(255, 255, 255, 0.15),
                    0 0 60px rgba(102, 126, 234, 0.1) !important;
        transform: translateY(-2px) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Streamlit Alert Override ── */
    .stAlert {
        background: rgba(102, 126, 234, 0.06) !important;
        border: 1px solid rgba(102, 126, 234, 0.15) !important;
        border-radius: 16px !important;
    }

    /* ── Processing Spinner ── */
    .processing-state {
        text-align: center;
        padding: 2rem;
    }

    .processing-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 1rem;
        animation: fadeIn 0.5s ease-out;
    }

    .spinner-ring {
        width: 48px;
        height: 48px;
        border: 2px solid rgba(102, 126, 234, 0.1);
        border-top-color: var(--accent-blue);
        border-radius: 50%;
        margin: 0 auto;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* ── Section Divider ── */
    .section-divider {
        width: 60px;
        height: 1px;
        background: var(--accent-gradient);
        margin: 2rem auto;
        border-radius: 1px;
    }

    /* ── How It Works ── */
    .how-it-works {
        display: flex;
        justify-content: center;
        gap: 2.5rem;
        margin: 1.5rem auto 2.5rem;
        max-width: 680px;
        flex-wrap: wrap;
    }

    .step {
        text-align: center;
        flex: 1;
        min-width: 120px;
    }

    .step-number {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--accent-blue);
    }

    .step-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 4px;
    }

    .step-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        color: var(--text-muted);
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">✦ AI-Powered Voice Analysis</div>
    <h1 class="hero-title">
        Feel the <span class="gradient-text">Frequency</span><br>
        of Every Emotion
    </h1>
    <p class="hero-subtitle">
        Aura listens to the subtle nuances in your voice and reveals the
        emotions hidden within — powered by advanced machine learning
        and real-time audio intelligence.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Stats Bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">8</div>
        <div class="stat-label">Emotions Detected</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">3s</div>
        <div class="stat-label">Recording Time</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">187</div>
        <div class="stat-label">Audio Features</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">~1s</div>
        <div class="stat-label">Analysis Speed</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── How It Works ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="how-it-works">
    <div class="step">
        <div class="step-number">01</div>
        <div class="step-title">Record</div>
        <div class="step-desc">Capture 3 seconds of your voice</div>
    </div>
    <div class="step">
        <div class="step-number">02</div>
        <div class="step-title">Analyze</div>
        <div class="step-desc">Extract MFCC, Chroma & Mel features</div>
    </div>
    <div class="step">
        <div class="step-number">03</div>
        <div class="step-title">Reveal</div>
        <div class="step-desc">AI classifies your emotional state</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Recording Section ────────────────────────────────────────────────────────
st.markdown("""
<div class="recording-card">
    <div class="card-header">
        <div class="card-icon">🎙️</div>
        <div>
            <div class="card-title">Voice Capture</div>
            <div class="card-desc">Press the button below to begin recording</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Centered button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    record_clicked = st.button("✦  Begin Recording", use_container_width=True)

if record_clicked:
    sr = 22050
    duration = 3

    # Single placeholder that gets replaced at each phase
    status_placeholder = st.empty()

    # ── Recording Phase ──
    status_placeholder.markdown("""
    <div class="recording-card" style="border-color: rgba(102, 126, 234, 0.3);">
        <div class="processing-state">
            <div class="waveform-container">
                {} 
            </div>
            <div class="processing-text">🔴 Recording in progress — speak naturally…</div>
        </div>
    </div>
    """.format(
        "".join([
            f'<div class="wave-bar active" style="animation-delay: {i * 0.05}s;"></div>'
            for i in range(40)
        ])
    ), unsafe_allow_html=True)

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    audio = audio.flatten()

    # ── Processing Phase (replaces recording) ──
    status_placeholder.markdown("""
    <div class="recording-card">
        <div class="processing-state">
            <div class="spinner-ring"></div>
            <div class="processing-text">Analyzing vocal patterns…</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    features = extract_features_from_audio(audio, sr=sr)
    features = features.reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)
    predicted_id = prediction[0]

    emotion, emoji, color, description = emotion_map.get(
        predicted_id, ("Unknown", "❓", "#94a3b8", "Unrecognized pattern")
    )

    # ── Result (replaces processing) ──
    status_placeholder.markdown(f"""
    <div class="result-card">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label">Detected Emotion</div>
        <div class="result-emotion" style="color: {color};">{emotion}</div>
        <div class="result-description">{description}</div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width: 85%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Emotion Spectrum ─────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

chips_html = ""
for eid, (name, emj, clr, desc) in emotion_map.items():
    chips_html += f"""
    <div class="emotion-chip">
        <span class="chip-emoji">{emj}</span>
        <span class="chip-label">{name}</span>
    </div>
    """

st.markdown(f"""
<div style="text-align:center; margin-bottom: 1rem;">
    <span style="font-family: 'Inter', sans-serif; font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted);">
        Emotion Spectrum
    </span>
</div>
<div class="emotion-grid">
    {chips_html}
</div>
""", unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="premium-footer">
    <div class="footer-brand">✦ Aura</div>
    <div class="footer-note">Emotion Intelligence · Powered by Machine Learning</div>
</div>
""", unsafe_allow_html=True)
