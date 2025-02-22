#pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn torch torchvision torchaudio torchsummary pillow tqdm

import pandas as pd
import random

def recommend(mood):
    # Load the dataset
    df = pd.read_csv("songs.csv")
    pd.set_option('display.max_colwidth', None)
    language = input("Enter H or h for Hindi and E or e for English songs : ")
    if language == "H" or language=="h":
        filtered_data = df[(df["Mood"]==mood) & (df["Language"]=="Hindi")]
        data= filtered_data[["Song Title","Spotify Link"]].sample(n=5)
        print(data)
    elif language == "E" or language=="e":
        filtered_data = df[(df["Mood"]==mood) & (df["Language"]=="English")]
        data= filtered_data[["Song Title","Spotify Link"]].sample(n=5)
        print(data)
    else:
        print("No such language available")