import torch
import cv2
import numpy as np
from torchvision import transforms
from model import CNN  # Import the CNN class

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(channels=1, num_classes=7)  # 7 emotion classes
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Emotion categories
categories = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]  # Extract face ROI
        face_pil = cv2.resize(face, (48, 48))  # Resize to match model input
        from PIL import Image

        # Convert NumPy array to PIL Image before applying transformations
        face_pil = Image.fromarray(face_pil)  # Convert to PIL Image
        face_tensor = transform(face_pil).unsqueeze(0).to(device)  # Apply transformations

        # Predict emotion
        with torch.no_grad():
            output = model(face_tensor)
            _, predicted = torch.max(output, 1)
            emotion = categories[predicted.item()]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display video feed
    cv2.imshow("Emotion Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
