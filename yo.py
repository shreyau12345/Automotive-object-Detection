
# Install:
# pip install streamlit ultralytics opencv-python pillow pandas

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Smart Driving Eligibility System",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#0f172a,#1e293b);
    color:white;
}

h1,h2,h3{
    color:#38bdf8;
}

.result{
    font-size:30px;
    font-weight:bold;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.title("🚗 Smart AI Automotive Object Detection System")

st.write("AI Based Vehicle & Driver Verification System")


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📌 Menu")

option = st.sidebar.selectbox(
    "Select Detection Mode",
    [
        "Image Detection",
        "Webcam Detection"
    ]
)

# =====================================================
# USER DETAILS
# =====================================================

st.subheader("👤 Driver Information")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Enter Name")

    age = st.number_input(
        "Enter Age",
        min_value=1,
        max_value=100,
        value=18
    )

with col2:
    gender = st.selectbox(
        "Select Gender",
        ["Male", "Female", "Other"]
    )

    license_no = st.text_input("Driving License Number")

# =====================================================
# LOAD YOLO MODEL
# =====================================================

model = YOLO("yolov8n.pt")

# =====================================================
# IMAGE DETECTION
# =====================================================

if option == "Image Detection":

    uploaded_file = st.file_uploader(
        "Upload Vehicle Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        img_array = np.array(image)

        # ============================================
        # YOLO DETECTION
        # ============================================

        results = model(img_array)

        annotated_frame = results[0].plot()

        st.image(
            annotated_frame,
            caption="Detected Vehicles",
            use_container_width=True
        )

        # ============================================
        # DETECTED OBJECTS
        # ============================================

        detected_objects = []

        for box in results[0].boxes:

            cls = int(box.cls[0])

            label = model.names[cls]

            detected_objects.append(label)

        st.subheader("📋 Detected Objects")

        st.write(detected_objects)

        # ============================================
        # VEHICLE COUNT
        # ============================================

        vehicle_count = 0

        for obj in detected_objects:

            if obj in ["car", "truck", "bus", "motorcycle"]:

                vehicle_count += 1

        st.success(f"🚘 Total Vehicles Detected: {vehicle_count}")
        
        # ============================================
        # DRIVER ELIGIBILITY
        # ============================================

        st.subheader("✅ Driver Verification")

        st.write(f"👤 Name: {name}")
        st.write(f"🎂 Age: {age}")
        st.write(f"🪪 License No: {license_no}")
        st.write(f"⚧ Gender: {gender}")

        if age >= 18:

            st.markdown(
                '<p class="result" style="color:lightgreen;">✅ Eligible For Driving</p>',
                unsafe_allow_html=True
            )

            st.success("Driving Permission Granted 🚗")

        else:

            st.markdown(
                '<p class="result" style="color:red;">❌ Not Eligible For Driving</p>',
                unsafe_allow_html=True
            )

            st.error("Driving Permission Denied")

        # ============================================
        # REPORT
        # ============================================

        data = {
            "Name": [name],
            "Age": [age],
            "Gender": [gender],
            "License No": [license_no],
            "Vehicles Detected": [vehicle_count],
            "Date": [datetime.now()]
        }

        df = pd.DataFrame(data)

        st.subheader("📄 Driver Report")

        st.dataframe(df)

        csv = df.to_csv(index=False)

        st.download_button(
            "⬇ Download Report",
            csv,
            "driver_report.csv",
            "text/csv"
        )

# =====================================================
# WEBCAM DETECTION
# =====================================================

elif option == "Webcam Detection":

    st.subheader("📷 Live Webcam Detection")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(1)

    while run:

        ret, frame = camera.read()

        if not ret:
            st.error("Camera Not Working")
            break

        # YOLO Detection
        results = model(frame)

        annotated_frame = results[0].plot()

        FRAME_WINDOW.image(
            annotated_frame,
            channels="BGR"
        )

    camera.release()

# =====================================================
# FOOTER
# =====================================================

st.write("---")

st.write("Made with ❤️ using Streamlit + YOLO + OpenCV")

# =====================================================
# VEHICLE SPEED DETECTION USING OPENCV
# =====================================================

# Install:
# pip install streamlit opencv-python ultralytics numpy

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Vehicle Speed Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚀 AI Vehicle Speed Detection System")

# =====================================================
# LOAD YOLO MODEL
# =====================================================

model = YOLO("yolov8n.pt")

# =====================================================
# START CAMERA
# =====================================================

run = st.checkbox("Start Speed Detection")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

# =====================================================
# VARIABLES
# =====================================================

prev_time = time.time()

prev_x = 0

PIXELS_PER_METER = 8

# =====================================================
# SPEED DETECTION
# =====================================================

while run:

    success, frame = camera.read()

    if not success:
        st.error("Camera Not Working")
        break

    # YOLO Detection
    results = model(frame)

    annotated_frame = results[0].plot()

    # ============================================
    # DETECT VEHICLES
    # ============================================

    for box in results[0].boxes:

        cls = int(box.cls[0])

        label = model.names[cls]

        # Detect Only Vehicles
        if label in ["car", "truck", "bus", "motorcycle"]:

            x1, y1, x2, y2 = box.xyxy[0]

            # Vehicle Center
            center_x = int((x1 + x2) / 2)

            # ====================================
            # TIME CALCULATION
            # ====================================

            current_time = time.time()

            time_diff = current_time - prev_time

            # ====================================
            # DISTANCE CALCULATION
            # ====================================

            distance_pixels = abs(center_x - prev_x)

            distance_meters = distance_pixels / PIXELS_PER_METER

            # ====================================
            # SPEED FORMULA
            # ====================================

            if time_diff > 0:

                speed = (distance_meters / time_diff) * 3.6

            else:

                speed = 0

            # ====================================
            # UPDATE VALUES
            # ====================================

            prev_x = center_x

            prev_time = current_time

            # ====================================
            # DISPLAY SPEED
            # ====================================

            cv2.putText(
                annotated_frame,
                f"Speed: {int(speed)} km/h",
                (int(x1), int(y1)-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

    # ============================================
    # SHOW FRAME
    # ============================================

    FRAME_WINDOW.image(
        annotated_frame,
        channels="BGR"
    )

camera.release()

# =====================================================
# PRELOADER / PREPARATION SCREEN CODE
# ADD THIS BEFORE st.title()
# =====================================================

import time

# =====================================================
# # LOADING SCREEN
# # =====================================================

with st.spinner("🚗 Preparing Smart AI Driving System..."):

    progress_bar = st.progress(0)

    for i in range(100):

        time.sleep(0.03)

        progress_bar.progress(i + 1)

# =====================================================
# SUCCESS MESSAGE
# =====================================================

st.success("✅ System Ready Successfully!")


# =====================================================
# ANIMATED BACKGROUND
# =====================================================

page_bg = """
<style>

.stApp {
background-image: linear-gradient(
to right,
#0f172a,
#1e293b,
#334155
);

background-size: 400% 400%;

animation: gradient 10s ease infinite;
}

@keyframes gradient {

0% {
background-position: 0% 50%;
}

50% {
background-position: 100% 50%;
}

100% {
background-position: 0% 50%;
}

}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

st.image(
    "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM2M0YzM5MjM5ZWQxMzIwYjY2MWM5ZDIxY2JhYzFlNGE3N2E4MzM3YiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/l0HlBO7eyXzSZkJri/giphy.gif",
    width=250
)

# =====================================================
# PROFESSIONAL SIDEBAR DESIGN
# ADD THIS AFTER st.set_page_config()
# =====================================================

# =====================================================
# SIDEBAR CSS
# =====================================================

st.markdown("""
<style>

/* Sidebar Background */
[data-testid="stSidebar"]{
    background: linear-gradient(
    180deg,
    #0f172a,
    #1e293b,
    #334155
    );
    
    border-right:2px solid #38bdf8;
}

/* Sidebar Text */
[data-testid="stSidebar"] *{
    color:white;
}

/* Sidebar Title */
.sidebar-title{
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:#38bdf8;
    margin-bottom:20px;
}

/* Sidebar Card */
.sidebar-card{
    background:#1e293b;
    padding:15px;
    border-radius:15px;
    margin-bottom:15px;
    box-shadow:0px 0px 10px #38bdf8;
}

/* Status Dot */
.dot{
    height:12px;
    width:12px;
    background-color:lime;
    border-radius:50%;
    display:inline-block;
    margin-right:8px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR HEADER
# =====================================================

st.sidebar.markdown("""
<div class="sidebar-title">
🚗 Smart AI System
</div>
""", unsafe_allow_html=True)

# # =====================================================
# # SIDEBAR IMAGE
# # =====================================================

# st.sidebar.image(
#     "https://c:police-traffic-stop-3354772.webp",
#     width=180
# )

# =====================================================
# SYSTEM INFO
# =====================================================

from datetime import datetime

current_time = datetime.now()

st.sidebar.markdown("""
<div class="sidebar-card">

### 🕒 System Information

</div>
""", unsafe_allow_html=True)

st.sidebar.write("📅 Date:", current_time.date())

st.sidebar.write("⏰ Time:", current_time.strftime("%H:%M:%S"))

# =====================================================
# FOOTER
# =====================================================

st.sidebar.markdown("""
---

<center>
Made with ❤️ using Streamlit
</center>

""", unsafe_allow_html=True)

# =====================================================
# INSTALL
# pip install streamlit ultralytics opencv-python numpy
# =====================================================

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Vehicle Speed Monitoring",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("🚗 AI Vehicle Speed Monitoring System")

st.write("Real-Time Vehicle Speed Detection Using YOLO")

# =====================================================
# LOAD YOLO MODEL
# =====================================================

model = YOLO("yolov8n.pt")

# =====================================================
# START CAMERA
# =====================================================

run = st.checkbox("▶ Start Speed Monitoring")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

# =====================================================
# VARIABLES
# =====================================================

prev_time = time.time()

prev_x = 0

PIXELS_PER_METER = 8

speed_limit = st.sidebar.slider(
    "🚦 Speed Limit",
    20,
    150,
    60
)

# =====================================================
# SPEED DETECTION
# =====================================================

while run:

    success, frame = camera.read()

    if not success:

        st.error("❌ Camera Not Working")

        break

    # YOLO Detection
    results = model(frame)

    annotated_frame = results[0].plot()

    # =================================================
    # DETECT VEHICLES
    # =================================================

    for box in results[0].boxes:

        cls = int(box.cls[0])

        label = model.names[cls]

        # Detect Vehicles Only
        if label in ["car", "truck", "bus", "motorcycle"]:

            x1, y1, x2, y2 = box.xyxy[0]

            center_x = int((x1 + x2) / 2)

            # =========================================
            # TIME DIFFERENCE
            # =========================================

            current_time = time.time()

            time_diff = current_time - prev_time

            # =========================================
            # DISTANCE
            # =========================================

            distance_pixels = abs(center_x - prev_x)

            distance_meters = distance_pixels / PIXELS_PER_METER

            # =========================================
            # SPEED FORMULA
            # =========================================

            if time_diff > 0:

                speed = (distance_meters / time_diff) * 3.6

            else:

                speed = 0

            # =========================================
            # UPDATE VALUES
            # =========================================

            prev_x = center_x

            prev_time = current_time

            # =========================================
            # SPEED ALERT
            # =========================================

            if speed > speed_limit:

                color = (0,0,255)

                alert = "Overspeed!"

            else:

                color = (0,255,0)

                alert = "Normal"

            # =========================================
            # DISPLAY SPEED
            # =========================================

            cv2.putText(
                annotated_frame,
                f"{int(speed)} km/h - {alert}",
                (int(x1), int(y1)-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            # =========================================
            # VEHICLE BOX
            # =========================================

            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )

    # =================================================
    # SHOW FRAME
    # =================================================

    FRAME_WINDOW.image(
        annotated_frame,
        channels="BGR"
    )

camera.release()

# =====================================================
# FOOTER
# =====================================================

st.write("---")

st.success("✅ AI Speed Monitoring Active")

# =========================================================
# LIVE MOVING VEHICLE TRACKING SYSTEM
# =========================================================
# INSTALL:
# pip install streamlit ultralytics opencv-python numpy
# =========================================================

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Live Moving Vehicle Tracking",
    page_icon="🚗",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🚗 Live Moving Vehicle Tracking System")

st.write("AI Based Real-Time Vehicle Tracking Using YOLO")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Tracking Control")

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.1,
    1.0,
    0.5
)

# =========================================================
# LOAD YOLO MODEL
# =========================================================

model = YOLO("yolov8n.pt")

# =========================================================
# START CAMERA
# =========================================================

run = st.checkbox("▶ Start Vehicle Tracking")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

# =========================================================
# TRACKING VARIABLES
# =========================================================

vehicle_id = 0

tracked_centers = {}

# =========================================================
# TRACKING LOOP
# =========================================================

while run:

    success, frame = camera.read()

    if not success:

        st.error("❌ Camera Not Working")

        break

    # =====================================================
    # YOLO DETECTION
    # =====================================================

    results = model(frame)

    for box in results[0].boxes:

        cls = int(box.cls[0])

        label = model.names[cls]

        score = float(box.conf[0])

        # =================================================
        # VEHICLE FILTER
        # =================================================

        if label in ["car", "truck", "bus", "motorcycle"]:

            if score > confidence:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # =============================================
                # CENTER POINT
                # =============================================

                center_x = int((x1 + x2) / 2)

                center_y = int((y1 + y2) / 2)

                # =============================================
                # SIMPLE TRACKING
                # =============================================

                matched_id = None

                for vid, pt in tracked_centers.items():

                    distance = np.sqrt(
                        (center_x - pt[0])**2 +
                        (center_y - pt[1])**2
                    )

                    if distance < 50:

                        matched_id = vid

                        break

                # =============================================
                # NEW VEHICLE
                # =============================================

                if matched_id is None:

                    vehicle_id += 1

                    matched_id = vehicle_id

                tracked_centers[matched_id] = (
                    center_x,
                    center_y
                )

                # =============================================
                # DRAW BOX
                # =============================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                # =============================================
                # TRACK ID
                # =============================================

                cv2.putText(
                    frame,
                    f"ID: {matched_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

                # =============================================
                # CENTER DOT
                # =============================================

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (0,0,255),
                    -1
                )

    # =====================================================
    # SHOW FRAME
    # =====================================================

    FRAME_WINDOW.image(
        frame,
        channels="BGR"
    )

camera.release()

# =========================================================
# FOOTER
# =========================================================

st.write("---")

st.success("✅ Live Vehicle Tracking Active")




