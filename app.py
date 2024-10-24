import streamlit as st
from keras.models import load_model
import cv2
import numpy as np

# Load the pre-trained model and face classifier
model = load_model('final_model.h5')
face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Labels and colors for the mask detection
labels_dict = {1: 'MASK', 0: 'NO MASK'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

# Streamlit app title
st.title("Live Mask Detection System")

# Streamlit sidebar controls and information
st.sidebar.markdown("## Controls")
st.sidebar.success(
    "This application uses a webcam to detect whether individuals are wearing masks or not. "
    "Once you click the 'Start Detection' button, the system will begin analyzing the video feed in real-time. "
    "It will highlight faces detected in the frame and indicate whether they are wearing a mask (in green) or not (in red)."
)

# Session state for controlling the webcam
if "run" not in st.session_state:
    st.session_state["run"] = False

# Start/Stop Buttons
start = st.sidebar.button("Start Detection")
stop = st.sidebar.button("Stop Detection")

# Toggle session state based on button clicks
if start:
    st.session_state["run"] = True
if stop:
    st.session_state["run"] = False

# Function to process the image frame
def process_frame(source):
    ret, img = source.read()
    if not ret:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        # Draw rectangles and labels on the image
        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return img

# Start the webcam video capture
source = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Main loop for handling the video stream
if st.session_state["run"]:
    stframe = st.empty()  # Create a placeholder for the video frames

    while st.session_state["run"]:
        frame = process_frame(source)
        if frame is None:
            st.write("Unable to capture video.")
            break

        # Convert the frame to RGB format and display it in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

else:
    st.info("Please click the 'Start Detection' button to begin mask detection.")

# Release the video source when done
source.release()
