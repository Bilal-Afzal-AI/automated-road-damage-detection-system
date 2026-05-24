from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path("models/baseline/best.pt")


st.set_page_config(
    page_title="Road Damage Detection System",
    page_icon="🛣️",
    layout="wide",
)


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


st.title("🛣️ Automated Road Damage Detection System")
st.write("Upload a road image and the AI model will detect cracks, potholes, and road damage.")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload road image",
    type=["jpg", "jpeg", "png"],
)

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    results = model.predict(
        source=temp_path,
        imgsz=640,
        conf=confidence,
    )

    result_image = results[0].plot()

    with col2:
        st.subheader("Detected Road Damage")
        st.image(result_image, use_container_width=True)

    st.subheader("Detection Summary")

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf_score = float(box.conf[0])

            st.write(f"✅ **{class_name}** — Confidence: `{conf_score:.2f}`")
    else:
        st.warning("No road damage detected in this image.")