import cv2
import numpy as np

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gender_model = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
age_model = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')

# Define age categories
age_categories = [
    '0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100'
]

def detect_faces(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found or cannot be loaded.")

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_data = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]

        # Preprocess face for gender prediction
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        gender = 'Male' if gender_preds[0][0] > gender_preds[0][1] else 'Female'

        # Preprocess face for age prediction
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict age
        age_model.setInput(blob)
        age_preds = age_model.forward()
        age_index = np.argmax(age_preds[0])
        age = age_categories[age_index]

        face_data.append({'gender': gender, 'age': age})

    return face_data

# Test the module
image_path = r'C:\Users\User\Desktop\VISION\jscode.jpg'  # Replace with the path to your image
try:
    detected_faces = detect_faces(image_path)
    for idx, face in enumerate(detected_faces):
        print(f"Face {idx+1}: Gender: {face['gender']}, Age: {face['age']}")
except FileNotFoundError as e:
    print(e)
