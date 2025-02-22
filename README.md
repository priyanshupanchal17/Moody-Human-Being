# Moody-Human-Being
An Emotion Based Music Recommender
Emotion-Based Music Recommender

📌 Project Overview

The Emotion-Based Music Recommender is an AI-powered system that suggests music based on the user's emotional state. By analyzing facial expressions, the system detects emotions and recommends songs that align with the user's mood.

🚀 Features

🎭 Emotion Detection: Uses facial recognition via live web cam to identify emotions.

🎶 Personalized Music Recommendations: Suggests songs based on detected emotions.

🔍 Integration with Music APIs: Fetches songs from platforms like Spotify, YouTube, or Apple Music.

🌐 Web & Mobile Compatibility: Can be deployed as a web app or mobile app.

🛠️ Technologies Used

Python, OpenCV, TensorFlow/Keras, Pytorch (for facial emotion recognition)
📌 How It Works

Emotion Detection:

Uses a webcam for facial emotion recognition.

Music Recommendation:

Matches detected emotions to a pre-defined set of songs.

Dataset : FER 2013 FROM KAGGLE ( https://www.kaggle.com/datasets/msambare/fer2013 )
User Experience:

Displays the recommended songs.

Required Libraries : pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn torch torchvision torchaudio torchsummary pillow tqdm

detection.py : Running this file will open an interface that will detect the emotion of the user as Angry , Fear , Disgust , Sad , Happy , Surprise or Neutral and is a fun experimentation 

final_mood.py : Running this file will track the users facial expressions for 1 minute and based on the most confident emotion will show the 7 emotions on web cam but categorise it into Happy Sad or Neutral and recommend songs based on the same from a pre defined dataset

The Model has achieved a test accuracy of 66%


Scalability 
1) Can be deployed as an application using Flask or Streamlit 
2) User activity throughout the day can be tracked both text and facial features to determine the mood
3) Songs can be customised based on the user criterion

My trained model on : https://github.com/priyanshupanchal17/Moody-Human-Being/releases/tag/emo_recognition
