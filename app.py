"""StyleMuse Studio v2: actual hair segmentation + photographic recoloring."""
from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import UnidentifiedImageError

from stylemuse_core import (
    PRESETS,
    get_model_path,
    load_image,
    make_mask_preview,
    recolor_hair,
    segment_hair,
    smooth_mask,
    to_png_bytes,
)

st.set_page_config(
    page_title="StyleMuse Studio • Natural Hair Color Try-On",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
      background: radial-gradient(circle at 86% 1%, rgba(192,89,122,.15), transparent 36%),
                  radial-gradient(circle at 0% 60%, rgba(112,94,166,.08), transparent 40%),
                  #101016;
      color:#f7f1f4;
    }
    [data-testid="stHeader"] {background: transparent;}
    .block-container {max-width: 1220px; padding-top:2.2rem; padding-bottom:5rem;}
    h1,h2,h3 {letter-spacing:-.04em!important;}
    div[data-testid="stImage"] img {border-radius:18px;}
    div[data-testid="stFileUploader"] {border:1px solid #514451;border-radius:18px;padding:8px;}
    .sm-kicker {font-size:.77rem;letter-spacing:.24em;font-weight:800;color:#e7a6c1;text-transform:uppercase;}
    .sm-headline {font-size:clamp(2.7rem,5.8vw,5.1rem);letter-spacing:-.065em;line-height:1.04;
                  font-weight:840;color:#fff;margin:.3rem 0 .9rem;}
    .sm-headline em {font-style:normal;color:#edaac7;}
    .sm-intro {color:#c9c0cb;max-width:690px;font-size:1.08rem;line-height:1.66;}
    .sm-tags {font-size:.88rem;color:#b8abb7;margin:1.2rem 0 2rem;}
    .sm-eyebrow {font-size:.85rem;color:#d7a5bd;letter-spacing:.12em;font-weight:800;margin-bottom:.7rem;}
    .sm-hint {font-size:.88rem;color:#b8adbb;line-height:1.6;}
    div.stButton button[kind="primary"], div.stDownloadButton button[kind="primary"] {
      background:linear-gradient(90deg,#d381a6,#e8b5c2);border:0;color:#1b101a;border-radius:12px;
      font-weight:800;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {border-color:#423740!important;border-radius:18px!important;}
    .sm-foot {margin-top:2rem;font-size:.82rem;color:#a79cac;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="sm-kicker">✦ StyleMuse / The color studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sm-headline">New color. <em>Still you.</em></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sm-intro">Try shades on your actual hair, not a cartoon wig. '
    'Our free hair-segmentation model identifies the hair in your photo and '
    'recolors it while retaining strands, shine, and natural shadows.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sm-tags">✦ No OpenAI key &nbsp; • &nbsp; ✦ No image credits &nbsp; • &nbsp; '
    '✦ Designed to leave your face unchanged</div>', unsafe_allow_html=True,
)

left, right = st.columns([1, 1.4], gap="large")
with left:
    st.markdown('<div class="sm-eyebrow">01 &nbsp; ADD A PHOTO</div>', unsafe_allow_html=True)
    tab_upload, tab_camera = st.tabs(["Upload", "Use camera"])
    with tab_upload:
        uploaded = st.file_uploader("Choose a portrait", type=["jpg", "jpeg", "png", "webp"], help="Choose a well-lit image that clearly shows your hair.")
    with tab_camera:
        captured = st.camera_input("Take a selfie")

    st.markdown('<div class="sm-eyebrow" style="margin-top:1.1rem">02 &nbsp; CHOOSE YOUR SHADE</div>', unsafe_allow_html=True)
    color_name = st.selectbox("Hair color", list(PRESETS), index=3)
    color = st.color_picker("Fine-tune color (optional)", PRESETS[color_name], key="custom_dye") if st.toggle("Custom hair color", value=False) else PRESETS[color_name]
    strength_pct = st.slider("Color intensity", min_value=20, max_value=100, value=91, step=1)
    lightness = st.slider("Adjust brightness", min_value=-40, max_value=40, value=0, step=1)
    refine = st.expander("Advanced controls", expanded=False)
    with refine:
        threshold = st.slider("Hair detection sensitivity", 0.05, 0.6, 0.21, 0.01, help="Lower: include more fine strands. Higher: exclude uncertain areas.")
        feather = st.slider("Edge softness", 0.0, 5.0, 2.3, 0.1)
        show_mask = st.checkbox("Preview detected hair region", value=False)
    st.markdown('<div class="sm-hint">Tip: good lighting, clear hair edges and a simple background produce the best result. Very dark hair may not lighten to platinum convincingly.</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="sm-eyebrow">03 &nbsp; LIVE PREVIEW</div>', unsafe_allow_html=True)
    chosen = uploaded if uploaded is not None else captured
    if chosen is None:
        st.info("Upload or take a portrait to start your realistic hair-color preview.")
        st.markdown("**Your actual hair — not a painted-on hairstyle.**")
        st.caption("No image is processed until you upload one. After upload, shade changes are instant because your hair mask is cached.")
    else:
        try:
            data = chosen.getvalue()
            photo = load_image(data)
            rgb = np.asarray(photo, dtype=np.uint8)

            @st.cache_resource(show_spinner=False)
            def model_location() -> str:
                return get_model_path()

            @st.cache_data(show_spinner=False, max_entries=12)
            def compute_hair_mask(image_bytes: bytes, model_path: str) -> np.ndarray:
                image = load_image(image_bytes)
                return segment_hair(np.asarray(image, dtype=np.uint8), model_path)

            with st.spinner("Detecting hair with the free hair-segmentation model…"):
                model = model_location()
                raw_mask = compute_hair_mask(data, model)
            hair_mask = smooth_mask(raw_mask, threshold=threshold, feather=feather)
            if float((hair_mask > .35).mean()) < 0.001:
                st.warning("Couldn't reliably detect hair in this image. Try a brighter front-facing portrait.")
            else:
                preview = recolor_hair(photo, hair_mask, color, strength_pct / 100, lightness)
                before, after = st.columns(2, gap="small")
                with before:
                    st.image(photo, caption="Original", use_container_width=True)
                with after:
                    st.image(preview, caption=f"{color_name} · {strength_pct}%", use_container_width=True)
                st.download_button(
                    "↓ Download my new hair color",
                    data=to_png_bytes(preview),
                    file_name="stylemuse-new-hair-color.png",
                    mime="image/png", type="primary", use_container_width=True,
                )
                if show_mask:
                    st.image(make_mask_preview(photo, hair_mask), caption="Pink indicates the area the model identified as hair.", use_container_width=True)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"We couldn't use that photo: {exc}")
        except Exception as exc:
            st.error("Hair detection couldn't start. Please try rebooting your Streamlit app.")
            st.caption(f"Diagnostic details for developer: {exc}")

st.divider()
st.markdown(
    '<div class="sm-foot">Powered by the open MediaPipe HairSegmenter model. '
    '<b>True scope:</b> realistic recoloring of your existing haircut, not new-haircut generation. '
    'No OpenAI usage or paid inference calls. Photos are processed in the Streamlit '
    'server session and are not intentionally stored by this app.</div>',
    unsafe_allow_html=True,
)
