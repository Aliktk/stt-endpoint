"""Streamlit demo front-end for the STT endpoint.

Talks to the FastAPI service over HTTP so the whole stack is exercised end to
end. Every network call degrades to a friendly message rather than a traceback.
"""

import json
import time

import requests
import streamlit as st

DEFAULT_API = "http://localhost:8000"
UPLOAD_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "aac"]

# Friendly label -> language code sent to the API ("" means auto-detect).
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


def fetch_providers(api_url: str) -> list[str] | None:
    """Return available providers, or None when the API is unreachable."""
    try:
        resp = requests.get(f"{api_url}/v1/providers", timeout=5)
        resp.raise_for_status()
        return resp.json().get("available", [])
    except requests.RequestException:
        return None


def request_transcript(api_url: str, name: str, blob: bytes, language: str) -> requests.Response:
    return requests.post(
        f"{api_url}/v1/transcribe",
        files={"file": (name, blob)},
        data={"language": language} if language else None,
        timeout=1800,
    )


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:05.2f}"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2rem; max-width: 820px; }
          .hero {
            background: linear-gradient(120deg, #6C5CE7 0%, #8E7BEF 45%, #48C6EF 100%);
            padding: 1.7rem 1.9rem; border-radius: 18px; margin-bottom: 1.2rem;
            box-shadow: 0 12px 34px rgba(108, 92, 231, 0.28);
          }
          .hero h1 { color: #fff; margin: 0; font-size: 2rem; letter-spacing: -0.5px; }
          .hero p { color: rgba(255,255,255,0.92); margin: 0.4rem 0 0; font-size: 1rem; }
          .status-bar {
            display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
            background: #161923; border: 1px solid #262A38; border-radius: 12px;
            padding: 0.7rem 1rem; margin-bottom: 1.1rem;
          }
          .status-bar .label { color: #8B93A7; font-size: 0.85rem; margin-right: 0.2rem; }
          .pill {
            display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px;
            font-size: 0.82rem; font-weight: 600;
          }
          .pill-live { background: rgba(46, 204, 113, 0.16); color: #6FE39F; }
          .pill-off  { background: rgba(231, 76, 60, 0.16); color: #FF8B7E; }
          .transcript-card {
            background: #161923; border: 1px solid #262A38; border-radius: 12px;
            padding: 1.15rem 1.35rem; line-height: 1.65; font-size: 1.05rem;
          }
          div[data-testid="stMetric"] {
            background: #161923; border: 1px solid #262A38; border-radius: 12px;
            padding: 0.7rem 0.9rem;
          }
          div[data-testid="stMetricValue"] { font-size: 1.3rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status(providers: list[str] | None) -> None:
    if providers is None:
        body = "<span class='pill pill-off'>● API offline</span>"
    elif providers:
        pills = "".join(f"<span class='pill pill-live'>● {p}</span>" for p in providers)
        body = f"<span class='label'>Providers online</span>{pills}"
    else:
        body = "<span class='pill pill-off'>● no providers available</span>"
    st.markdown(f"<div class='status-bar'>{body}</div>", unsafe_allow_html=True)


def render_result(data: dict, elapsed: float, source_name: str) -> None:
    st.divider()
    a, b, c, d = st.columns(4)
    a.metric("Provider", data["provider"])
    b.metric("Language", data["language"])
    c.metric("Duration", f"{data['duration']:.1f}s")
    d.metric("Elapsed", f"{elapsed:.1f}s")

    st.subheader("Transcript")
    st.markdown(
        f"<div class='transcript-card'>{data['text'] or '<em>(no speech detected)</em>'}</div>",
        unsafe_allow_html=True,
    )

    st.subheader(f"Segments · {len(data['segments'])}")
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


def main() -> None:
    inject_styles()
    st.markdown(
        "<div class='hero'><h1>🎙️ Speech-to-Text Pipeline</h1>"
        "<p>API-first transcription with automatic local fallback and per-segment timestamps.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Advanced settings"):
        api_url = st.text_input("API URL", value=DEFAULT_API).rstrip("/")

    render_status(fetch_providers(api_url))

    uploaded = st.file_uploader("Upload an audio file", type=UPLOAD_TYPES)
    if uploaded is None:
        st.caption("Supported formats: " + ", ".join(UPLOAD_TYPES).upper())
        return

    st.audio(uploaded)
    picker, action = st.columns([2, 1])
    label = picker.selectbox("Language", list(LANGUAGES), index=0)
    action.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    go = action.button("Transcribe", type="primary", use_container_width=True)
    if not go:
        return

    started = time.time()
    with st.spinner("Transcribing… larger files are chunked automatically."):
        try:
            resp = request_transcript(api_url, uploaded.name, uploaded.getvalue(), LANGUAGES[label])
        except requests.RequestException as exc:
            st.error(f"Could not reach the API at {api_url}. Is it running? ({exc})")
            return

    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.content else resp.reason
        st.error(f"Transcription failed: {detail}")
        return

    render_result(resp.json(), time.time() - started, uploaded.name)


main()
