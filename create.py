import os
import pickle
import mediapipe as mp
import cv2

# Inisialisasi Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Daftar label dari 0 hingga 23
labels_list = list(range(24))  # Menggunakan angka dari 0 hingga 23

# Loop melalui setiap direktori (kelas)
for label in labels_list:
    dir_path = os.path.join(DATA_DIR, str(label))  # Menggunakan label sebagai nama direktori
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} does not exist. Skipping.")
        continue

    img_files = os.listdir(dir_path)
    if not img_files:
        print(f"No images found in directory {dir_path}. Skipping.")
        continue

    for img_path in img_files:
        data_aux = []

        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            print(f"Error: Could not read image {img_path}. Skipping.")
            continue  # Coba lagi jika gagal membaca gambar

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Proses gambar untuk mendeteksi landmark tangan
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []

                # Ambil koordinat landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalisasi dan simpan data landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Pastikan data_aux memiliki panjang yang diharapkan
                if len(data_aux) == 42:  # 21 landmarks * 2 (x dan y)
                    data.append(data_aux)
                    labels.append(label)  # Menyimpan label sebagai angka
                else:
                    print(f"Warning: Data for {img_path} has unexpected length ({len(data_aux)}). Skipping.")
        else:
            print(f"Warning: No hands detected in {img_path}. Skipping.")

# Cek apakah data berhasil dikumpulkan
if not data or not labels:
    print("Error: No valid data collected. Please check the images and try again.")
else:
    # Simpan data dan label ke file pickle
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Data and labels have been saved successfully.")
    print("pppp")
