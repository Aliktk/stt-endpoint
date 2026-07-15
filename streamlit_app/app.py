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

st.set_page_config(page_title="STT Endpoint", page_icon="🎙️", layout="wide")


def fetch_providers(api_url: str) -> list[str]:
    try:
        resp = requests.get(f"{api_url}/v1/providers", timeout=5)
        resp.raise_for_status()
        return resp.json().get("available", [])
    except requests.RequestException:
        return []


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
          .block-container { padding-top: 2rem; max-width: 1120px; }
          .hero {
            background: linear-gradient(120deg, #6C5CE7 0%, #8E7BEF 45%, #48C6EF 100%);
            padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.4rem;
            box-shadow: 0 10px 30px rgba(108, 92, 231, 0.25);
          }
          .hero h1 { color: #fff; margin: 0; font-size: 1.9rem; letter-spacing: -0.5px; }
          .hero p { color: rgba(255,255,255,0.9); margin: 0.35rem 0 0; font-size: 0.98rem; }
          .pill {
            display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
            background: rgba(108,92,231,0.18); color: #C4B9FF; font-size: 0.82rem;
            font-weight: 600; margin: 0.15rem 0.3rem 0.15rem 0;
          }
          .pill-live { background: rgba(46, 204, 113, 0.16); color: #6FE39F; }
          .transcript-card {
            background: #161923; border: 1px solid #262A38; border-radius: 12px;
            padding: 1.1rem 1.3rem; line-height: 1.65; font-size: 1.02rem;
          }
          div[data-testid="stMetricValue"] { font-size: 1.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.header("⚙️ Settings")
        api_url = st.text_input("API URL", value=DEFAULT_API).rstrip("/")
        language = st.text_input("Language code", value="", placeholder="auto-detect")
        st.divider()
        st.subheader("Providers online")
        available = fetch_providers(api_url)
        if available:
            st.markdown(
                "".join(f"<span class='pill pill-live'>● {p}</span>" for p in available),
                unsafe_allow_html=True,
            )
        else:
            st.warning("API unreachable — start it with `uv run uvicorn app.main:app`.")
    return api_url, language


def render_result(data: dict, elapsed: float, source_name: str) -> None:
    left, mid, right = st.columns(3)
    left.metric("Provider", data["provider"])
    mid.metric("Language", data["language"])
    right.metric("Elapsed", f"{elapsed:.1f}s")

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

    col_json, col_txt = st.columns(2)
    col_json.download_button(
        "⬇️ Download JSON",
        data=json.dumps(data, indent=2),
        file_name=f"{source_name}.transcript.json",
        mime="application/json",
        use_container_width=True,
    )
    col_txt.download_button(
        "⬇️ Download text",
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

    api_url, language = render_sidebar()

    uploaded = st.file_uploader("Drop an audio file to transcribe", type=UPLOAD_TYPES)
    if uploaded is None:
        st.info("Supported formats: " + ", ".join(UPLOAD_TYPES).upper())
        return

    st.audio(uploaded)
    if not st.button("Transcribe", type="primary", use_container_width=True):
        return

    started = time.time()
    with st.spinner("Transcribing… larger files are chunked automatically."):
        try:
            resp = request_transcript(api_url, uploaded.name, uploaded.getvalue(), language)
        except requests.RequestException as exc:
            st.error(f"Could not reach the API at {api_url}. Is it running? ({exc})")
            return

    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.content else resp.reason
        st.error(f"Transcription failed: {detail}")
        return

    render_result(resp.json(), time.time() - started, uploaded.name)


main()
