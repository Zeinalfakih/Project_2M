import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check if data and labels are not empty
if len(data) == 0 or len(labels) == 0:
    print("Error: No data found. Please check the data collection process.")
    exit()

# Preprocess the labels
labels = to_categorical(labels, num_classes=24)  # Specify num_classes as 24 (your number of classes)

# Inspect the shape of the data
print(f"Data shape: {data.shape}")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Build the DNN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(42,)),  # Input layer with 42 features (flattened data)
    Dense(64, activation='relu'),  # Hidden layer with 64 units
    Dense(32, activation='relu'),  # Hidden layer with 32 units
    Dense(24, activation='softmax')  # Output layer with 24 units (one for each class)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print accuracy
print(f'{score[1] * 100:.2f}% of samples were classified correctly!')

# Save the model
model.save('dnn_model.h5')
