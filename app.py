import streamlit as st
import cv2
import numpy as np
import av
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Load the pre-trained model and face classifier
MODEL_PATH = 'final_model.h5'
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Load model and compile if needed
model = load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

face_clsfr = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Labels and colors for mask detection
LABELS_DICT = {1: 'MASK', 0: 'NO MASK'}
COLOR_DICT = {1: (0, 255, 0), 0: (0, 0, 255)}

# Streamlit app title
st.title("Real-Time Mask Detection with Streamlit WebRTC")

# Sidebar instructions
st.sidebar.markdown("## Controls")
st.sidebar.info("This system uses your webcam to detect whether individuals are wearing masks in real-time.")

# Define a custom VideoProcessor for WebRTC
class MaskDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize with the pre-loaded model and classifier
        self.model = model
        self.face_clsfr = face_clsfr

    def recv(self, frame):
        # Get the image frame from the video stream
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_clsfr.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Preprocess the face region for the model
            face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(face_img, (100, 100))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 100, 100, 1))  # Ensure this matches the model's input shape
            
            # Try-catch block to handle potential prediction errors
            try:
                result = self.model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # Draw rectangles and labels on the image
            color = COLOR_DICT[label]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(img, (x, y - 40), (x + w, y), color, -1)
            cv2.putText(img, LABELS_DICT[label], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Return the processed frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC streamer setup using the custom processor
webrtc_streamer(
    key="mask-detection",
    video_processor_factory=MaskDetectionProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)
