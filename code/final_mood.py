
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import time
from model import CNN
from recommendation import recommend

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

# Emotion mapping
def filter_emotions(emotion):
    """Maps detected emotions into three primary categories while keeping UI display intact"""
    emotion_mapping = {
        "Angry": "Sad",
        "Fear": "Sad",
        "Disgust": "Sad",
        "Surprise": "Happy",
        "Happy": "Happy",
        "Neutral": "Neutral",
        "Sad": "Sad",
    }
    
    # Apply mapping
    mapped_emotion = emotion_mapping[emotion]

    # Count occurrences of rare emotions
    rare_count = sum(1 for e, _ in emotion_history if e == emotion)

    # Ensure rare emotions are stable before allowing them
    if emotion in ["Angry", "Fear", "Disgust", "Surprise"] and rare_count < 3:
        mapped_emotion = "Neutral"  # Default to neutral if rare emotions are infrequent

    return emotion, mapped_emotion  # Return both original and mapped emotions

# Function to compute final mood
def get_final_mood():
    """Computes the most frequent mapped emotion from history"""
    if not emotion_history:
        return "Neutral"  # Default if no face detected
    
    mood_counts = {"Happy": 0, "Sad": 0, "Neutral": 0}
    
    for mapped_emotion, confidence in emotion_history:
        mood_counts[mapped_emotion] += confidence  # Weight by confidence score
    
    # Return mood with the highest confidence sum
    return max(mood_counts, key=mood_counts.get)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion history storage
emotion_history = []
start_time = time.time()
duration = 30  # Run for 1 minute

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_pil = Image.fromarray(cv2.resize(face, (48, 48)))

        # Convert to tensor and predict
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
            predicted_index = np.argmax(probabilities)
            raw_emotion = categories[predicted_index]  # Raw detected emotion
            confidence_score = probabilities[predicted_index]

        # **Apply emotion filtering & mapping**
        original_emotion, mapped_emotion = filter_emotions(raw_emotion)

        # **Store both original & mapped emotion**
        emotion_history.append((mapped_emotion, confidence_score))

        # **Display original emotion in OpenCV UI**
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, original_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display video feed
    cv2.imshow("Emotion Recognition", frame)

    # Stop after 1 minute
    if time.time() - start_time > duration:
        break

    # Press 'q' to exit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Get final mood after 1 minute
final_mood = get_final_mood()
print(f"Final User Mood: {final_mood}")
# Release resources
cap.release()
cv2.destroyAllWindows()

recommend(final_mood)



