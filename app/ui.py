import os, sys, tempfile, re
import streamlit as st
import plotly.graph_objects as go


# Ensure local imports work when running "streamlit run app/app.py"
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from transcriber import transcribe_audio
from evaluator import evaluate_transcript

# --- Optional LLM feedback (only if you want it) ---
USE_LLM_DEFAULT = False
try:
    from feedback import generate_supportive_feedback  # your optional module
    HAS_LLM = True
except Exception:
    HAS_LLM = False

# ----- Helper: score gauge -----
def render_score_gauge(score: int):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2563eb"},
            "steps": [
                {"range": [0, 60], "color": "#ef4444"},   # red
                {"range": [60, 80], "color": "#f59e0b"},  # amber
                {"range": [80, 100], "color": "#22c55e"}, # green
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=10))
    return fig

# ----- Helper: label chips (emoji + color) -----
def chip(text: str, color_hex: str, emoji: str):
    st.markdown(
        f"""
        <span style="
            display:inline-block;
            padding:6px 12px;
            border-radius:9999px;
            background:{color_hex}1A;
            color:{color_hex};
            font-weight:600;
            margin-right:8px;
        ">{emoji} {text}</span>
        """,
        unsafe_allow_html=True
    )

# ----- Bucketing logic for user-friendly labels -----
def bucket_filler(ratio: float):
    if ratio < 0.03:  return ("Low", "#22c55e", "✅")
    if ratio < 0.07:  return ("Medium", "#f59e0b", "🟡")
    return ("High", "#ef4444", "😬")

def bucket_clarity(ratio: float):
    if ratio > 0.55:  return ("Clear", "#22c55e", "📝")
    if ratio > 0.40:  return ("Moderate", "#f59e0b", "🟠")
    return ("Low", "#ef4444", "⚠️")

def bucket_sentiment(compound: float):
    if compound >= 0.60:  return ("Positive", "#22c55e", "🙂")
    if compound >= 0.20:  return ("Neutral",  "#f59e0b", "😐")
    return ("Negative", "#ef4444", "🙁")


# -------------- Page config --------------
st.set_page_config(
    page_title="Interview Evaluator",
    page_icon="🎤",
    layout="wide",
)

st.title("🎤 Interview Evaluator — MVP")
st.caption("Upload an interview answer audio, transcribe it, and get instant, actionable feedback.")

# -------------- Sidebar --------------
st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox(
    "Whisper model",
    ["base", "small", "medium"],
    index=0,
    help="Heavier models are more accurate but slower."
)

strip_question = st.sidebar.checkbox(
    "Strip interviewer's opening question (heuristic)",
    value=True,
    help="Removes the first sentence if it ends with '?'."
)

enable_llm = st.sidebar.checkbox(
    "Generate supportive LLM feedback (optional)",
    value=USE_LLM_DEFAULT and HAS_LLM,
    help="Uses your feedback module. Requires valid API key set inside that file or as env var."
)

if enable_llm and not HAS_LLM:
    st.sidebar.warning("LLM feedback module not found (feedback_generator.py). UI will run without it.")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: For best results, record a 30–60s answer with a clear voice and minimal background noise.")

# -------------- Upload --------------
uploaded = st.file_uploader("Upload audio file (.mp3 / .wav / .m4a)", type=["mp3", "wav", "m4a"])

col_btn1, col_btn2 = st.columns([1,1])
with col_btn1:
    run_btn = st.button("▶️ Evaluate")
with col_btn2:
    ex_btn = st.button("Use example audio (skips upload)")

# -------------- Helpers --------------
def _strip_leading_question(text: str) -> str:
    qpos = text.find("?")
    if 0 <= qpos < 200:
        return text[qpos+1:].strip()
    return text

# -------------- Main Flow --------------
if (uploaded or ex_btn) and run_btn:
    # Save file to temp
    if ex_btn and not uploaded:
        st.info("No example file bundled — please upload your own MP3/WAV. (Or place a sample in code and wire it here.)")
        st.stop()

    suffix = os.path.splitext(uploaded.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Transcribe
    with st.spinner("Transcribing audio with Whisper…"):
        # If you want to honor model_choice, you can expose it in transcriber.py;
        # for now, we set env var the transcriber can read, or you can modify transcriber to accept model
        os.environ["WHISPER_MODEL_NAME"] = model_choice
        transcript = transcribe_audio(tmp_path)

    if strip_question:
        transcript = _strip_leading_question(transcript)

    st.subheader("📝 Transcript")
    st.write(transcript if transcript.strip() else "_(empty transcript)_")

    # Evaluate
    with st.spinner("Analyzing clarity, fillers, and tone…"):
        result = evaluate_transcript(transcript)

    if "error" in result:
        st.error(result["error"])
        st.stop()

    # ----------- Metrics Layout -----------
    st.subheader("📊 Evaluation")

    # ---- Big Circular Gauge ----
    center = st.columns([1, 2, 1])[1]
    with center:
        st.markdown("### 🎯 Communication Score")
        fig = render_score_gauge(int(result["scores"]["final"]))
        st.plotly_chart(fig, use_container_width=False)

    # ---- Short labels (emoji + color) ----
    st.markdown("### 🔎 Quick View")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        st.metric("Total Words", result["total_words"])
    with c2:
        f_lbl, f_col, f_emo = bucket_filler(result["filler_ratio"])
        st.caption("Filler Usage")
        chip(f_lbl, f_col, f_emo)
    with c3:
        c_lbl, c_col, c_emo = bucket_clarity(result["clarity_ratio"])
        st.caption("Clarity Level")
        chip(c_lbl, c_col, c_emo)
    with c4:
        s_lbl, s_col, s_emo = bucket_sentiment(result["sentiment"]["compound"])
        st.caption("Tone")
        chip(s_lbl, s_col, s_emo)

    
    # No sub-scores needed — already represented as High/Low labels
    st.markdown("---")


    # ---- Details (numbers tucked away) ----
    with st.expander("Show technical details (ratios & raw)"):
        st.write(f"Filler Ratio: **{result['filler_ratio']}**")
        st.write(f"Clarity Ratio: **{result['clarity_ratio']}**")
        st.write(f"Sentiment: **{result['sentiment']['label']}** ({round(result['sentiment']['compound'], 3)})")


    st.markdown("#### System Feedback")
    st.info(result["feedback"])

    # Optional: show raw JSON (useful for debugging / viva)
    with st.expander("See raw evaluation JSON"):
        st.json(result)

    # LLM Feedback (optional)
    if enable_llm and HAS_LLM:
        st.markdown("#### 💬 Supportive Coaching Feedback")
        try:
            with st.spinner("Generating supportive feedback…"):
                llm_text = generate_supportive_feedback(transcript, result)
            st.success(llm_text)
        except Exception as e:
            st.warning(f"LLM feedback failed: {e}")

    # Cleanup
    try:
        os.remove(tmp_path)
    except Exception:
        pass

elif run_btn and not uploaded and not ex_btn:
    st.warning("Please upload an audio file first.")
else:
    st.caption("Upload an audio file and click **Evaluate** to begin.")
