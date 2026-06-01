import streamlit as st
import av
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import pandas as pd
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Smart AI Driving Eligibility System",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>
.stApp{
background:linear-gradient(to right,#0f172a,#1e293b);
color:white;
}

h1,h2,h3{
color:#38bdf8;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.title("🚗 Smart AI Driving Eligibility System")
st.write("AI Based Vehicle & Driver Verification System")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📌 Menu")

option = st.sidebar.selectbox(
    "Select Mode",
    [
        "Image Detection",
        "Live Camera Detection"
    ]
)

# =====================================================
# DRIVER INFO
# =====================================================

st.subheader("👤 Driver Information")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Name")
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=18
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    license_no = st.text_input(
        "License Number"
    )

# =====================================================
# IMAGE DETECTION
# =====================================================

if option == "Image Detection":

    uploaded_file = st.file_uploader(
        "Upload Vehicle Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        img = np.array(image)

        results = model(img)

        detected_img = results[0].plot()

        st.image(
            detected_img,
            caption="Detected Objects",
            use_container_width=True
        )

        detected_objects = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detected_objects.append(label)

        st.subheader("Detected Objects")
        st.write(detected_objects)

        vehicle_count = sum(
            1 for obj in detected_objects
            if obj in [
                "car",
                "truck",
                "bus",
                "motorcycle"
            ]
        )

        st.success(
            f"🚘 Vehicles Detected: {vehicle_count}"
        )

        # Driver Verification

        st.subheader("Driver Verification")

        if age >= 18:
            st.success(
                "✅ Eligible For Driving"
            )
        else:
            st.error(
                "❌ Not Eligible For Driving"
            )

        report = pd.DataFrame({
            "Name":[name],
            "Age":[age],
            "Gender":[gender],
            "License":[license_no],
            "Vehicles":[vehicle_count],
            "Date":[datetime.now()]
        })

        st.dataframe(report)

        csv = report.to_csv(index=False)

        st.download_button(
            "Download Report",
            csv,
            "driver_report.csv",
            "text/csv"
        )

# =====================================================
# LIVE CAMERA DETECTION
# =====================================================

elif option == "Live Camera Detection":

    st.subheader("📷 Live Camera Detection")

    confidence = st.slider(
        "Confidence",
        0.1,
        1.0,
        0.5,
        0.05
    )

    class YOLOProcessor(VideoProcessorBase):

        def recv(self, frame):

            img = frame.to_ndarray(
                format="bgr24"
            )

            results = model.predict(
                img,
                conf=confidence,
                verbose=False
            )

            annotated = results[0].plot()

            return av.VideoFrame.from_ndarray(
                annotated,
                format="bgr24"
            )

    webrtc_streamer(
        key="vehicle-detection",
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.info(
        "Allow browser camera permission and click START."
    )

# =====================================================
# FOOTER
# =====================================================

st.write("---")
st.success("🚀 Smart AI Driving System Ready")
