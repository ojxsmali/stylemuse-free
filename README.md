# StyleMuse Studio v2 — No API, Real Hair Color

A separate, free Streamlit website which uses **Google's MediaPipe HairSegmenter** to detect the actual hair in portraits and tint it while maintaining photographic texture.

## What improved

- **Removes cartoon wig overlays entirely.** Recolors existing hair only.
- Hair isolation from a free ML segmentation model; soft edge blending.
- 12 color presets and optional custom color, intensity, lightness, feather and sensitivity controls.
- Upload/camera, original/preview, PNG save, optional hair-mask debug display.
- No OpenAI credentials, no inference API account, and no per-image API charges.

## Important limitation

It **cannot create realistic new hairstyles/haircuts**. The shape of the hair remains exactly the same. Photorealistic haircut transfer/generation needs a separate image-generation model and usually a GPU, which Streamlit Community Cloud does not guarantee. Brightening black hair to platinum is also approximate.

## Deploy to your existing GitHub repository

Repo: `https://github.com/ojxsmali/stylemuse-free`

1. In your repo, replace existing **`app.py`** and **`requirements.txt`** with the files in this package (GitHub `Add file` → `Upload files` can replace existing names).
2. Upload the new **`stylemuse_core.py`** file as well.
3. Optionally upload the `.streamlit/config.toml` file (GitHub web upload may hide folders; optional).
4. Commit changes on `main`. Your `stylemuse-free.streamlit.app` should auto-redeploy.
5. Watch **Manage app → Logs** on Streamlit if it fails. On first use the model downloads (under 1 MB) from Google's public server and is cached. No API key is used. Allow outbound model download on first run.

You can leave the old `hairmuse_core.py` in your repo; this upgrade doesn't import it.

## Local run

Python 3.12 recommended:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works

Official model URL: https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/1/hair_segmenter.tflite

The app attempts to fetch a public `.tflite` model once, then runs it on server CPU via the local MediaPipe Python library. This is a **model asset download**, not a billable inference API call. Images are not uploaded to any AI endpoint by the code, although photos necessarily pass through Streamlit's hosting infrastructure. Free-hosting resource and usage restrictions still apply.

## Tests

```bash
python -m unittest discover -s tests -v
```
