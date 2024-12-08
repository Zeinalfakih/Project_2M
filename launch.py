import pickle
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st

# Memuat model DNN yang sudah dilatih
model = load_model('dnn_model.h5')

# Mendefinisikan label untuk karakter tangan (dari 0 sampai 23 untuk A sampai Y)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
               12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X',
               23: 'Y'}

# Inisialisasi MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Streamlit UI Setup
st.title("Hand Gesture Recognition")
st.write("Use your webcam to make hand gestures. The model will recognize the gesture in real-time.")

# Tombol Start dan Stop
start_button = st.button("Start Video Stream")
stop_button = st.button("Stop Video Stream")

# Create a placeholder for the webcam video stream
frame_placeholder = st.empty()

# Flag untuk memastikan hanya satu kali release webcam
is_streaming = False
cap = None

# Mulai Streaming Video Jika Tombol "Start" Ditekan
if start_button and not is_streaming:
    is_streaming = True
    # Start video capture with OpenCV
    cap = cv2.VideoCapture(0)

    # Check if webcam is available
    if not cap.isOpened():
        st.error("Error: Cannot access webcam.")
        cap.release()
        exit()

    # Loop untuk menangkap dan memproses video
    while is_streaming:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe to detect hand landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,  # frame to draw landmarks on
                    hand_landmarks,  # model output landmarks
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                data_aux = []
                x_ = []
                y_ = []

                # Collect hand landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize the landmark data
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalize x
                    data_aux.append(y - min(y_))  # Normalize y

                # Ensure the data matches the model's expected input size (42)
                if len(data_aux) == 42:
                    input_data = np.array([data_aux])
                    input_data = input_data.astype(np.float32)

                    # Predict using the model
                    prediction = model.predict(input_data)
                    predicted_class = np.argmax(prediction)
                    predicted_character = labels_dict[predicted_class]

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

        # Check if Stop button was pressed and stop streaming
        if stop_button:
            is_streaming = False

    # Clean up after the loop stops
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    frame_placeholder.empty()

