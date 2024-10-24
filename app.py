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

start = st.sidebar.button("Start Detection")
stop = st.sidebar.button("Stop Detection")

# Start the webcam video capture
source = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Function to process the image frame
def process_frame():
    ret, img = source.read()
    if not ret:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]  # Corrected for height
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

# Main loop for handling the video stream
if start:
    stframe = st.empty()  # Create a placeholder for the video frames

    while True:
        frame = process_frame()
        if frame is None:
            st.write("Unable to capture video.")
            break

        # Convert the frame to RGB format and display it in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Exit loop when stop button is clicked
        if stop:
            break

else:
    st.info("Please click the 'Start Detection' button to begin mask detection.")

# Release the video source when done
source.release()
