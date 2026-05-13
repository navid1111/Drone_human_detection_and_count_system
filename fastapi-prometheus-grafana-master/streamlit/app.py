import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def fetch_image_bytes(url: str) -> bytes | None:
    try:
        img_resp = requests.get(url, timeout=30)
        if img_resp.status_code == 200:
            return img_resp.content
    except requests.RequestException:
        return None
    return None

st.set_page_config(page_title="Retail Detection Dashboard", layout="wide")
st.title("Retail Object Detection")
st.caption("Upload an image, run detection, and track model outputs over time.")

with st.sidebar:
    st.header("API Settings")
    api_url = st.text_input("FastAPI URL", value=API_URL)

left, right = st.columns([2, 1])

with left:
    st.subheader("Run Detection")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    gt_text = st.text_area(
        "Optional ground-truth JSON",
        placeholder='{"labels": [{"class": "bottle", "bbox": [10, 20, 100, 220]}]}',
        height=120,
    )

    if st.button("Predict", use_container_width=True):
        if uploaded is None:
            st.warning("Please upload an image first.")
        else:
            files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")}
            data = {}
            if gt_text.strip():
                data["ground_truth"] = gt_text.strip()

            with st.spinner("Running inference..."):
                try:
                    resp = requests.post(f"{api_url}/predict", files=files, data=data, timeout=120)
                except requests.RequestException as exc:
                    st.error(f"Backend unavailable: {exc}")
                    st.stop()

            if resp.status_code != 200:
                st.error(f"Prediction failed ({resp.status_code}): {resp.text}")
            else:
                result = resp.json()
                st.success(f"Prediction #{result['id']} completed in {result['inference_ms']:.2f} ms")

                annotated_url = f"{api_url}{result['annotated_image_url']}"
                original_url = f"{api_url}{result['image_url']}"
                original_bytes = fetch_image_bytes(original_url)
                annotated_bytes = fetch_image_bytes(annotated_url)

                img_cols = st.columns(2)
                with img_cols[0]:
                    st.markdown("**Original**")
                    if original_bytes is not None:
                        st.image(original_bytes, use_container_width=True)
                    else:
                        st.error("Could not load original image from backend.")
                with img_cols[1]:
                    st.markdown("**Detections**")
                    if annotated_bytes is not None:
                        st.image(annotated_bytes, use_container_width=True)
                    else:
                        st.error("Could not load annotated image from backend.")

                if result["detections"]:
                    st.markdown("**Detection Details**")
                    st.dataframe(pd.DataFrame(result["detections"]), use_container_width=True)
                else:
                    st.info("No detections found.")

with right:
    st.subheader("Recent Predictions")
    try:
        hist = requests.get(f"{api_url}/predictions?limit=20", timeout=20)
        if hist.status_code == 200:
            rows = hist.json()
            if rows:
                table_rows = [
                    {
                        "id": r["id"],
                        "created_at": r["created_at"],
                        "model": r["model_name"],
                        "inference_ms": round(r["inference_ms"], 2),
                        "detections": len(r["detections"]),
                    }
                    for r in rows
                ]
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
            else:
                st.info("No predictions logged yet.")
        else:
            st.error(f"Could not load history: {hist.status_code}")
    except requests.RequestException as exc:
        st.error(f"Backend unavailable: {exc}")
