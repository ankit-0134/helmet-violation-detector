import streamlit as st
import cv2
import numpy as np
import os
import time
import tempfile
import subprocess
from PIL import Image
import pandas as pd
import io

from models.bike_person_detector import BikePersonDetector
from models.helmet_detector import HelmetDetector
from utils.video_processor import process_frame
from utils.violation_handler import ViolationHandler

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Helmet Violation Detector",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0d1117; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #f0f6fc !important; }
    .stat-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-value { font-size: 2rem; font-weight: 700; color: #f0f6fc; }
    .violation-badge {
        display: inline-block;
        background: #da3633;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("**Model Weights**")
    bike_weights   = st.text_input("Bike / Person model (.pt)", value="weights/bike_person.pt")
    helmet_weights = st.text_input("Helmet / LP model (.pt)",   value="weights/helmet_lp.pt")

    st.divider()
    st.markdown("**Detection Thresholds**")
    bike_conf    = st.slider("Bike/Person confidence",       0.10, 0.95, 0.40, 0.05)
    helmet_conf  = st.slider("Helmet confidence",            0.10, 0.95, 0.40, 0.05)
    size_thresh  = st.slider("Min bike size (% of frame)",   0.5,  10.0, 2.0,  0.5) / 100.0

    st.divider()
    st.markdown("**Processing**")
    frame_skip   = st.slider("Process every N frames",        1, 5,  1)
    display_fps  = st.slider("Display FPS cap",               5, 30, 15)
    cooldown     = st.slider("Violation save cooldown (frames)", 10, 90, 30, 5)

    st.divider()
    if st.button("🗑️ Clear All Violations", use_container_width=True):
        st.session_state.violation_handler = ViolationHandler()
        st.session_state.total_violations  = 0
        st.session_state.total_frames      = 0
        st.success("Cleared!")

    st.divider()
    st.markdown("""
    **Color Legend**
    🟠 Bike &nbsp;&nbsp; 🔵 Person  
    🟢 Helmet &nbsp; 🔴 No Helmet  
    🟣 License Plate
    """)

# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
if "violation_handler" not in st.session_state:
    st.session_state.violation_handler = ViolationHandler()
if "total_violations" not in st.session_state:
    st.session_state.total_violations = 0
if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0


# ──────────────────────────────────────────────
# Load models (cached so they don't reload on every rerun)
# ──────────────────────────────────────────────
@st.cache_resource
def load_models(bw, hw, bc, hc, st_val):
    bike_det   = BikePersonDetector(bw, conf_threshold=bc, size_threshold=st_val)
    helmet_det = HelmetDetector(hw, conf_threshold=hc)
    return bike_det, helmet_det


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("# 🪖 Helmet Violation Detection")
st.caption("Upload a traffic video — real-time detection of riders not wearing helmets")

# ──────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────
col_video, col_panel = st.columns([3, 1], gap="medium")

with col_panel:
    st.markdown("### 📊 Live Stats")
    stat_frames     = st.empty()
    stat_violations = st.empty()
    stat_rate       = st.empty()
    st.divider()
    st.markdown("### 🚨 Recent Violations")
    violations_table = st.empty()

with col_video:
    video_placeholder = st.empty()
    progress_placeholder = st.empty()

# ──────────────────────────────────────────────
# File uploader
# ──────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded:
    # ── Validate weights ──
    missing = []
    if not os.path.exists(bike_weights):
        missing.append(f"`{bike_weights}`")
    if not os.path.exists(helmet_weights):
        missing.append(f"`{helmet_weights}`")
    if missing:
        st.error(f"❌ Weight file(s) not found: {', '.join(missing)}\n\nDrop your `.pt` files into the `weights/` folder.")
        st.stop()

    # ── Load models ──
    with st.spinner("🔧 Loading models..."):
        try:
            bike_det, helmet_det = load_models(
                bike_weights, helmet_weights, bike_conf, helmet_conf, size_thresh
            )
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            st.stop()

    # ── Save uploaded video to temp file ──
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.flush()

    cap          = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Setup output video writer ──
    raw_out_temp = tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4")
    raw_out_path = raw_out_temp.name
    raw_out_temp.close()

    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(raw_out_path, fourcc, fps_video, (vid_w, vid_h))

    vh             = st.session_state.violation_handler
    cooldown_dict  = {}
    frame_idx      = 0
    delay          = 1.0 / display_fps

    stop_btn = st.button("⏹️ Stop", key="stop_processing")

    bike_tracker = {}
    next_bike_id = [1]

    # ── Main processing loop ──
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_btn:
            break

        frame_idx += 1
        st.session_state.total_frames = frame_idx

        if frame_idx % frame_skip != 0:
            continue

        annotated, violation = process_frame(
          frame, bike_det, helmet_det, vh,
          frame_idx, cooldown_dict,
          cooldown_frames=cooldown,
          bike_tracker=bike_tracker,
          next_bike_id=next_bike_id,
          fps=fps_video,
          )

        # ── Write annotated frame to output video ──
        out_writer.write(annotated)

        if violation:
            st.session_state.total_violations += 1

        # Display
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb, channels="RGB", use_container_width=True)

        # Progress bar
        pct = frame_idx / max(total_frames, 1)
        progress_placeholder.progress(
            min(pct, 1.0),
            text=f"Frame {frame_idx} / {total_frames}  |  {pct*100:.1f}%"
        )

        # Stats panel
        stat_frames.metric("Frames Processed", f"{st.session_state.total_frames:,}")
        stat_violations.metric("Violations Found", st.session_state.total_violations)
        elapsed_sec = frame_idx / fps_video
        rate = (st.session_state.total_violations / max(elapsed_sec, 1)) * 60
        stat_rate.metric("Violations / min", f"{rate:.1f}")

        # Recent violations table
        if vh.records:
            rows = [
                {
                    "#":      r.id,
                    "Frame":  r.frame_number,
                    "Plate":  "✅" if r.plate_path else "—",
                }
                for r in reversed(vh.records[-8:])
            ]
            violations_table.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        time.sleep(delay)

    cap.release()
    out_writer.release()

    # ── Re-encode with H.264 for broad browser/player compatibility ──
    final_out_temp = tempfile.NamedTemporaryFile(delete=False, suffix="_output.mp4")
    final_out_path = final_out_temp.name
    final_out_temp.close()

    ffmpeg_available = False
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", raw_out_path,
                "-vcodec", "libx264",
                "-crf", "23",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                final_out_path,
            ],
            check=True,
            capture_output=True,
        )
        ffmpeg_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg not installed — fall back to the raw mp4v file
        final_out_path = raw_out_path

    # ── Done banner ──
    st.success(
        f"✅ Processing complete — {frame_idx:,} frames scanned, "
        f"**{st.session_state.total_violations} violation(s)** detected."
    )

    # ── Download processed video ──
    try:
        with open(final_out_path, "rb") as f:
            video_bytes = f.read()

        st.download_button(
            label="⬇️ Download Processed Video",
            data=video_bytes,
            file_name="helmet_violation_output.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

        if not ffmpeg_available:
            st.caption(
                "⚠️ `ffmpeg` not found — downloaded file uses `mp4v` codec. "
                "Install ffmpeg (`sudo apt install ffmpeg`) for H.264 output "
                "with better compatibility."
            )
    except Exception as e:
        st.warning(f"Could not prepare video for download: {e}")

    # ── Cleanup temp files ──
    try:
        os.unlink(tfile.name)
        if ffmpeg_available:
            os.unlink(raw_out_path)
    except Exception:
        pass

    # ──────────────────────────────────────────
    # Gallery
    # ──────────────────────────────────────────
    if vh.records:
        st.divider()
        st.markdown(f"## 🖼️ Violation Gallery  `{len(vh.records)} total`")

        cols_per_row = 3
        records = list(reversed(vh.records))   # newest first

        for row_start in range(0, len(records), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, record in zip(cols, records[row_start: row_start + cols_per_row]):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**Violation #{record.id}** — Frame `{record.frame_number}`")
                        if os.path.exists(record.frame_path):
                            st.image(
                                Image.open(record.frame_path),
                                use_container_width=True,
                            )
                        if record.plate_path and os.path.exists(record.plate_path):
                            st.image(
                                Image.open(record.plate_path),
                                caption="🔢 License Plate Crop",
                                use_container_width=True,
                            )
                        else:
                            st.caption("🔢 No license plate detected")

        # ── Download CSV ──
        df = pd.DataFrame([
            {
                "Violation ID": r.id,
                "Frame Number": r.frame_number,
                "Timestamp":    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.timestamp)),
                "Frame Saved":  r.frame_path,
                "Plate Saved":  r.plate_path or "N/A",
            }
            for r in vh.records
        ])

        buf = io.BytesIO()
        df.to_csv(buf, index=False)

        st.download_button(
            label="⬇️ Download Violations CSV",
            data=buf.getvalue(),
            file_name="violations.csv",
            mime="text/csv",
        )

else:
    # Placeholder when no video uploaded
    with col_video:
        video_placeholder.markdown("""
        <div style="
            height: 360px;
            border: 2px dashed #30363d;
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #8b949e;
            font-size: 1.1rem;
        ">
            <div style="font-size: 3rem">🎥</div>
            <div style="margin-top: 12px">Upload a video above to begin</div>
        </div>
        """, unsafe_allow_html=True)