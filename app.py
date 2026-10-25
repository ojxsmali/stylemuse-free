"""StyleMuse Studio: a standalone, no-API-key hairstyle try-on app."""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image, UnidentifiedImageError

from hairmuse_core import (
    COLORS,
    STYLES,
    FaceBox,
    detect_face,
    prepare_image,
    render_preview,
    sample_portrait,
    to_png_bytes,
)

st.set_page_config(
    page_title="StyleMuse Studio | Free Hairstyle Try-On",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp {
        background: radial-gradient(ellipse at 80% 0%, #302139 0%, transparent 40%),
                    linear-gradient(145deg, #120f1b, #211825 60%, #181821);
      }
      .block-container {padding-top: 2.1rem; max-width: 1120px;}
      h1, h2, h3 {letter-spacing: -0.045em;}
      .hero-eyebrow {letter-spacing: 0.2em; color: #eda6b9; font-size: 0.73rem;
                     font-weight: 700; text-transform: uppercase;}
      .hero-title {color: #fff; font-size: clamp(2.5rem, 5.5vw, 4.2rem);
                   line-height: 1.03; font-weight: 850; letter-spacing: -0.06em;
                   margin: .25rem 0 1rem;}
      .hero-title b {color: #f7abc3;}
      .hero-subtitle {max-width: 720px; color: #d6cbdc; font-size: 1.05rem;
                      line-height: 1.6; margin-bottom: 1.45rem;}
      .small-note {color: #b4aabd; font-size: 0.85rem; line-height: 1.55;}
      .features {color: #e7dce8; font-size: .95rem; margin: 1.25rem 0;}
      div[data-testid="stImage"] img {border-radius: 16px;}
      div[data-testid="stSidebar"] {background: #1a1623;}
      div[data-testid="stFileUploader"] {border-radius: 12px;}
      div[data-testid="stAlert"] {border-radius: 12px;}
      .stDownloadButton button {font-weight: 750;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-eyebrow">Your personal hairstyle playground</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">Meet your next <b>look.</b></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">Try new hairstyles and colors with your own photo. '
    'Free to use — no account, no paid API key, no image-generation credits.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="features">✦ 8 hairstyle illustrations &nbsp;&nbsp; '
    '✦ 9 hair colors &nbsp;&nbsp; ✦ Camera or photo upload &nbsp;&nbsp; '
    '✦ Download your new look</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("✂️ StyleMuse Studio")
    st.caption("Personalize your look")
    st.divider()
    style = st.selectbox("Choose a hairstyle", STYLES, index=1)
    color_name = st.selectbox("Choose a hair color", list(COLORS), index=1)
    st.markdown(
        '<div style="height:16px;border-radius:99px;background:'
        + COLORS[color_name]
        + ';border:1px solid #ffffff35"></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Fine-tune the fit**")
    offset_x = st.slider("Move left / right", -70, 70, 0, step=2)
    offset_y = st.slider("Move up / down", -70, 70, 0, step=2)
    size_percent = st.slider("Hairstyle size (%)", 60, 170, 100, step=5)
    st.caption("The app estimates face placement locally. Adjust the sliders if needed.")
    st.divider()
    st.caption("🛡️ Your photo is not sent to OpenAI or a generative-image API.")

st.markdown("### 01 / Upload your photo")
st.caption("Use a well-lit, front-facing selfie for the best placement.")
upload_tab, camera_tab = st.tabs(["📁 Upload a photo", "📷 Take a photo"])
with upload_tab:
    uploaded_file = st.file_uploader(
        "Choose JPG, PNG or WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        help="Uploaded photos are processed only within the Streamlit app session.",
    )
with camera_tab:
    taken_photo = st.camera_input("Use your device camera")

try:
    if uploaded_file is not None:
        photo = prepare_image(Image.open(io.BytesIO(uploaded_file.getvalue())))
        sample = False
    elif taken_photo is not None:
        photo = prepare_image(Image.open(io.BytesIO(taken_photo.getvalue())))
        sample = False
    else:
        photo = sample_portrait()
        sample = True
except (OSError, UnidentifiedImageError, ValueError):
    st.error("We couldn't open that photo. Try uploading a JPG or PNG image.")
    st.stop()

# An internal illustration is shown until the user supplies a photo.
face: FaceBox = FaceBox(215, 235, 290, 435, True) if sample else detect_face(photo)

result = render_preview(
    photo,
    style,
    COLORS[color_name],
    face,
    offset_x=offset_x,
    offset_y=offset_y,
    size_percent=size_percent,
)

st.markdown("### 02 / Your new look")
if sample:
    st.info("Demo mode: upload your own photo to try the hairstyle on yourself.")
elif not face.detected:
    st.warning("We couldn't confidently find a face. Use the adjustment sliders to position the hair.")

original_col, styled_col = st.columns(2, gap="medium")
with original_col:
    st.image(photo, caption="Before", use_container_width=True)
with styled_col:
    st.image(result, caption=f"After · {style} in {color_name}", use_container_width=True)

st.markdown("### 03 / Save your look")
st.download_button(
    "⬇ Download your hairstyle preview",
    data=to_png_bytes(result),
    file_name="stylemuse-hair-preview.png",
    mime="image/png",
    type="primary",
    use_container_width=True,
)
st.markdown(
    '<p class="small-note">This is an <b>illustrated hairstyle overlay</b>, not '
    'photorealistic AI hair replacement. All image processing happens in this '
    'app’s runtime, with no OpenAI integration. Your image still passes through '
    'your website’s hosting infrastructure.</p>',
    unsafe_allow_html=True,
)
