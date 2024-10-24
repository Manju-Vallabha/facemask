import streamlit as st
import numpy as np
import av
import cv2
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# Load the pre-trained mask detection model
model = load_model('final_model.h5')

# Load the Haar cascade classifier for face detection
face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Labels and colors for mask detection
labels_dict = {1: 'MASK', 0: 'NO MASK'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

# Streamlit app title and layout
st.set_page_config(layout='wide', page_title="Face Mask Detection", page_icon="😷")
st.title("Real-Time Face Mask Detection")

# Sidebar instructions
st.sidebar.markdown("## Instructions")
st.sidebar.info("This application detects whether people are wearing face masks in real-time using your webcam. "
                "Faces with masks will be highlighted in green, and faces without masks in red.")

# Function for real-time mask detection using the webcam
def mask_detection_callback(frame):
    image_np = frame.to_ndarray(format="bgr24")  # Convert video frame to numpy array
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]  # Extract the face area from the image
        resized = cv2.resize(face_img, (100, 100))  # Resize the face for model input
        normalized = resized / 255.0  # Normalize the pixel values
        reshaped = np.reshape(normalized, (1, 100, 100, 1))  # Reshape to fit model input
        result = model.predict(reshaped)  # Predict whether mask is present

        label = np.argmax(result, axis=1)[0]  # Get the prediction (MASK or NO MASK)

        # Draw rectangles and labels on the image
        cv2.rectangle(image_np, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(image_np, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(image_np, labels_dict[label], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return av.VideoFrame.from_ndarray(image_np, format="bgr24")

# WebRTC component for webcam streaming and real-time mask detection
webrtc_streamer(key="mask_detection", video_frame_callback=mask_detection_callback,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ))

# Display instructions on the main page
st.markdown("""
    ### How to use:
    - Allow access to your webcam by clicking the 'Allow' button in the pop-up.
    - The system will start detecting faces and determining if they are wearing a mask in real-time.
    - Faces with masks will be highlighted in green, and those without masks in red.
""")
