import streamlit as st
import tensorflow as tf
import pandas as pd
import av
import cv2
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from PIL import Image, ImageDraw, ImageFont
from keras.models import load_model

# Set page configurations
st.set_page_config(layout='wide', page_title="Face Mask Detection", page_icon="😷")

# Load pre-trained mask detection model and other assets
model = load_model('final_model.h5')
face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define label and color for mask detection
labels_dict = {1: 'MASK', 0: 'NO MASK'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

# Face mask detection function
def web_mask_detection(frame):
    image_np = frame.to_ndarray(format="bgr24")
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)
    faces, confidences = cv.detect_face(image_np)
    
    for f in faces:
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        face_img = image_np[startY:endY, startX:endX]
        face_img_resized = cv.resize(face_img, (100, 100))
        face_img_normalized = face_img_resized / 255.0
        face_img_reshaped = np.reshape(face_img_normalized, (1, 100, 100, 3))
        
        # Predict mask or no mask
        result = model.predict(face_img_reshaped)
        label = np.argmax(result, axis=1)[0]

        # Draw rectangles and labels
        draw.rectangle(((startX, startY), (endX, endY)), outline=color_dict[label], width=2)
        label_text = labels_dict[label]
        font = ImageFont.truetype("arial.ttf", 15)
        draw.text((startX + 10, startY - 20), label_text, font=font, fill=(255, 255, 255))

    image_np = np.array(image_pil)
    return av.VideoFrame.from_ndarray(image_np, format="bgr24")

# Streamlit Sidebar
with st.sidebar:
    st.title('Face Mask Detection')
    st.success('This is a real-time face mask detection application using a webcam.')
    st.info('The model detects if individuals are wearing a mask or not.')
    task2 = ['<Select>', 'Capture', 'Web-Cam']
    mode = st.selectbox('Select Mode', task2)

# Main app layout
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-weight: bold;
        }
        .mainheading {
            text-align: center;
            font-family: monospace;
            font-size: 25px;
        }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<h1 class="title">Face Mask Detection</h1>', unsafe_allow_html=True)
st.markdown('<br><br>', unsafe_allow_html=True)

# Capture Mode
if mode == 'Capture':
    image = st.camera_input("Capture a snapshot")
    if image is not None:
        image = Image.open(image)
        image_np = np.array(image)
        image_pil = Image.fromarray(image_np)
        draw = ImageDraw.Draw(image_pil)
        faces, confidences = cv.detect_face(image_np)
        
        for f in faces:
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            face_img = image_np[startY:endY, startX:endX]
            face_img_resized = cv.resize(face_img, (100, 100))
            face_img_normalized = face_img_resized / 255.0
            face_img_reshaped = np.reshape(face_img_normalized, (1, 100, 100, 3))
            
            result = model.predict(face_img_reshaped)
            label = np.argmax(result, axis=1)[0]
            
            draw.rectangle(((startX, startY), (endX, endY)), outline=color_dict[label], width=2)
            label_text = labels_dict[label]
            font = ImageFont.truetype("arial.ttf", 15)
            draw.text((startX + 10, startY - 20), label_text, font=font, fill=(255, 255, 255))
        
        st.image(np.array(image_pil), caption='Captured Image with Face Mask Detection', use_column_width=True)

# Webcam Mode
if mode == 'Web-Cam':
    webrtc_streamer(key="example", video_frame_callback=web_mask_detection,
                    rtc_configuration=RTCConfiguration(
                        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                    ))
