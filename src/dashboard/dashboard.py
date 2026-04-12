import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from loguru import logger

logger.add(
    "streamlit.log",
    rotation="10 MB",
    retention="10 days",
    level="DEBUG",
    format="{time} | {level} | {message}"
)


def dashboard(output: str | Path) -> None:
    """Fire and Smoke Monitoring Dashboard"""

    inferred_location = Path(output)

    if not inferred_location.exists():
        st.error(f"Output location doesn't exist: {inferred_location}")
        return

    st.set_page_config(layout="wide")
    st.title("Fire and Smoke Monitoring Dashboard")

    # LOAD EVENTS
    events_path = inferred_location / "json" / "events.json"
    events: List[Dict[str, Any]] = []

    if events_path.exists():
        try:
            with open(events_path, "r") as f:
                events = json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON: {e}")
            st.error("Failed to load events.json")
            return

    # SUMMARY
    st.header("Summary")

    severity_counts: Dict[str, int] = {}

    for e in events:
        severity = e.get("severity", "UNKNOWN")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CRITICAL", severity_counts.get("CRITICAL", 0))
    col2.metric("HIGH", severity_counts.get("HIGH", 0))
    col3.metric("MEDIUM", severity_counts.get("MEDIUM", 0))
    col4.metric("LOW", severity_counts.get("LOW", 0))


    # SIDEBAR FILTER
    st.sidebar.header("Controls")

    severity_options = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]

    selected_severity = st.sidebar.selectbox(
        "Select Severity",
        severity_options
    )

    filtered_events = [
        e for e in events
        if selected_severity == "ALL" or e.get("severity") == selected_severity
    ]


    # LOAD MEDIA

    image_loc = inferred_location / "images"
    video_loc = inferred_location / "videos"
    images = sorted(image_loc.glob("*.jpg"), reverse=True)
    videos = sorted(video_loc.glob("*_h264.mp4"), reverse=True)

    filtered_images = [
        img for img in images
        if selected_severity == "ALL" or img.name.startswith(selected_severity)
    ]

    filtered_videos = [
        vid for vid in videos
        if selected_severity == "ALL" or vid.name.startswith(selected_severity)
    ]

    # SIDEBAR MEDIA SELECT
    st.sidebar.subheader("Frames")

    selected_image = None
    if filtered_images:
        selected_image_name = st.sidebar.selectbox(
            "Select Image",
            [img.name for img in filtered_images]
        )
        selected_image = image_loc / selected_image_name

    st.sidebar.subheader("Clips")

    selected_video = None
    if filtered_videos:
        selected_video_name = st.sidebar.selectbox(
            "Select Video",
            [vid.name for vid in filtered_videos]
        )
        selected_video = (video_loc / selected_video_name).resolve()
        st.write("Selected video:", selected_video)

    # MAIN DISPLA
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🖼️ Selected Frame")

        if selected_image:
            st.image(str(selected_image), caption=selected_image.name)
        else:
            st.info("No image selected")

    with col_right:
        st.subheader("🎥 Selected Clip")

        if selected_video:
            video_file = open(selected_video, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
        else:
            st.info("No video selected")

    # EVENTS TABLE
    st.header("Events")

    if not filtered_events:
        st.info("No events found")
    else:
        df = pd.DataFrame(filtered_events)

        st.dataframe(df, use_container_width=True, height=300)

        st.download_button(
            label="⬇️ Export as CSV",
            data=df.to_csv(index=False),
            file_name="events.csv",
            mime="text/csv"
        )

    # AUTO REFRESH
    st.markdown("---")

    if st.checkbox("Auto-refresh (every 5 sec)"):
        time.sleep(5)
        st.rerun()


# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "Inference")
    )

    args = parser.parse_args()

    dashboard(output=args.output)
