"""Streamlit demo front-end for the STT endpoint.

The UI runs the transcription pipeline in-process by importing the ``app``
package directly — no separate API server needed. The FastAPI service is a
second, independent way to use the very same pipeline.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# Make the project root importable when launched as `streamlit run …`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.errors import TranscriptionError  # noqa: E402
from app.providers.registry import build_providers  # noqa: E402
from app.services.transcription_service import TranscriptionService  # noqa: E402

UPLOAD_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "aac"]

# Friendly label -> language code passed to the pipeline ("" means auto-detect).
LANGUAGES = {
    "Auto-detect": "",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Hindi": "hi",
    "Urdu": "ur",
    "Arabic": "ar",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
}

st.set_page_config(page_title="STT Endpoint", page_icon="🎙️", layout="centered")


@st.cache_resource(show_spinner=False)
def get_service() -> TranscriptionService:
    """Build the pipeline once and reuse it across reruns.

    Caching keeps the local whisper model loaded instead of reloading it on
    every interaction.
    """
    settings = get_settings()
    return TranscriptionService(
        providers=build_providers(settings),
        chunk_threshold_seconds=settings.chunk_threshold_seconds,
        max_upload_mb=settings.max_upload_mb,
    )


def transcribe_upload(uploaded, language: str) -> dict:
    """Persist the upload to a temp file and run it through the pipeline."""
    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        temp_path = Path(tmp.name)
    try:
        result = get_service().transcribe(temp_path, language or None)
        return result.model_dump()
    finally:
        temp_path.unlink(missing_ok=True)


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:05.2f}"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

          html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
          .block-container { padding-top: 2rem; max-width: 820px; }

          @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
          }
          @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(14px); }
            to   { opacity: 1; transform: translateY(0); }
          }
          @keyframes pulseGlow {
            0%   { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.45); }
            70%  { box-shadow: 0 0 0 7px rgba(46, 204, 113, 0); }
            100% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
          }

          .hero {
            background: linear-gradient(120deg, #6C5CE7, #8E7BEF, #48C6EF, #6C5CE7);
            background-size: 300% 300%;
            animation: gradientShift 12s ease infinite;
            padding: 1.8rem 2rem; border-radius: 20px; margin-bottom: 1.2rem;
            box-shadow: 0 14px 40px rgba(108, 92, 231, 0.35);
          }
          .hero h1 {
            color: #fff; margin: 0; font-size: 2.1rem; font-weight: 800;
            letter-spacing: -0.6px;
          }
          .hero p { color: rgba(255,255,255,0.94); margin: 0.45rem 0 0; font-size: 1.02rem; }

          .status-bar {
            display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
            background: rgba(22, 25, 35, 0.6); backdrop-filter: blur(8px);
            border: 1px solid #262A38; border-radius: 14px;
            padding: 0.7rem 1rem; margin-bottom: 1.1rem;
          }
          .status-bar .label { color: #8B93A7; font-size: 0.85rem; margin-right: 0.2rem; }
          .pill {
            display: inline-block; padding: 0.28rem 0.8rem; border-radius: 999px;
            font-size: 0.82rem; font-weight: 600;
          }
          .pill-live {
            background: rgba(46, 204, 113, 0.16); color: #6FE39F;
            animation: pulseGlow 2.4s infinite;
          }
          .pill-off { background: rgba(231, 76, 60, 0.16); color: #FF8B7E; }

          .results { animation: fadeInUp 0.5s ease both; }
          .section-title {
            font-size: 1.15rem; font-weight: 700; margin: 1.1rem 0 0.4rem;
            background: linear-gradient(90deg, #C4B9FF, #7FD8F0);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          }
          .transcript-card {
            background: rgba(22, 25, 35, 0.7); backdrop-filter: blur(10px);
            border: 1px solid #2C3142; border-radius: 14px;
            padding: 1.2rem 1.4rem; line-height: 1.7; font-size: 1.06rem;
          }
          div[data-testid="stMetric"] {
            background: rgba(22, 25, 35, 0.7); backdrop-filter: blur(10px);
            border: 1px solid #2C3142; border-radius: 14px;
            padding: 0.75rem 0.95rem; transition: transform 0.15s ease, border-color 0.15s ease;
          }
          div[data-testid="stMetric"]:hover {
            transform: translateY(-3px); border-color: #6C5CE7;
          }
          div[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; }

          .stButton > button {
            border-radius: 12px; font-weight: 600; transition: all 0.18s ease;
          }
          .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 22px rgba(108, 92, 231, 0.4);
          }
          .stDownloadButton > button { border-radius: 12px; transition: all 0.18s ease; }
          .stDownloadButton > button:hover { transform: translateY(-2px); }

          .app-footer {
            text-align: center; color: #5B6172; font-size: 0.82rem;
            margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #20242F;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status(providers: list[str]) -> None:
    if providers:
        pills = "".join(f"<span class='pill pill-live'>● {p}</span>" for p in providers)
        body = f"<span class='label'>Providers online</span>{pills}"
    else:
        body = "<span class='pill pill-off'>● no providers available</span>"
    st.markdown(f"<div class='status-bar'>{body}</div>", unsafe_allow_html=True)


def render_result(data: dict, elapsed: float, source_name: str) -> None:
    st.markdown("<div class='results'>", unsafe_allow_html=True)
    st.divider()
    a, b, c, d = st.columns(4)
    a.metric("Provider", data["provider"])
    b.metric("Language", data["language"])
    c.metric("Duration", f"{data['duration']:.1f}s")
    d.metric("Elapsed", f"{elapsed:.1f}s")

    st.markdown("<div class='section-title'>Transcript</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='transcript-card'>{data['text'] or '<em>(no speech detected)</em>'}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='section-title'>Segments · {len(data['segments'])}</div>",
        unsafe_allow_html=True,
    )
    rows = [
        {
            "start": format_timestamp(s["start"]),
            "end": format_timestamp(s["end"]),
            "text": s["text"],
        }
        for s in data["segments"]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    left.download_button(
        "⬇️  Download JSON",
        data=json.dumps(data, indent=2),
        file_name=f"{source_name}.transcript.json",
        mime="application/json",
        use_container_width=True,
    )
    right.download_button(
        "⬇️  Download text",
        data=data["text"],
        file_name=f"{source_name}.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    st.markdown(
        "<div class='hero'><h1>🎙️ Speech-to-Text Pipeline</h1>"
        "<p>API-first transcription with automatic local fallback and per-segment timestamps.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    render_status(get_service().provider_names)

    uploaded = st.file_uploader("Upload an audio file", type=UPLOAD_TYPES)
    if uploaded is None:
        st.caption("Supported formats: " + ", ".join(UPLOAD_TYPES).upper())
        return

    st.audio(uploaded)
    picker, action = st.columns([2, 1])
    label = picker.selectbox("Language", list(LANGUAGES), index=0)
    action.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    if not action.button("Transcribe", type="primary", use_container_width=True):
        return

    started = time.time()
    with st.spinner("Transcribing… larger files are chunked automatically."):
        try:
            data = transcribe_upload(uploaded, LANGUAGES[label])
        except TranscriptionError as exc:
            st.error(f"Transcription failed: {exc}")
            return

    render_result(data, time.time() - started, uploaded.name)
    if data["text"]:
        st.toast("Transcription complete!", icon="✅")
        st.balloons()


def render_footer() -> None:
    st.markdown(
        "<div class='app-footer'>Powered by ElevenLabs · Deepgram · faster-whisper "
        "— runs the FastAPI pipeline in-process</div>",
        unsafe_allow_html=True,
    )


main()
render_footer()
