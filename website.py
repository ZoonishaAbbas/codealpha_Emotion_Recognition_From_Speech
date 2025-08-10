import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib

# === Page Config ===
st.set_page_config(page_title="Emotion from Speech 🎧", page_icon="🎙️", layout="centered")

# === CSS Styling & Animated Background ===
st.markdown("""
    <style>
    html, body, .stApp {
        height: 100%;
        margin: 0;
        background: linear-gradient(-45deg, #89f7fe, #66a6ff, #e0c3fc, #8ec5fc);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .emotion-box {
        padding: 1rem;
        background-color: #e3f2fd;
        border-left: 6px solid #2196f3;
        border-radius: 10px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }

    .music-box {
        background-color: #fff3e0;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 6px solid #ff9800;
        margin-top: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }

    .music-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.6rem;
        color: #ff6f00;
    }

    .stButton>button {
        background-color: #6c63ff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        transition: 0.3s ease;
        margin-top: 10px;
    }

    .stButton>button:hover {
        background-color: #4e45b3;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# === Load Model and Label Encoder ===
@st.cache_resource
def load_artifacts():
    model = load_model("emotion_model.keras")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder

model, le = load_artifacts()

# === Emotion Prediction Function ===
def predict_emotion(file_path, model, label_encoder, max_pad_len=174):
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        prediction = model.predict(mfcc)
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        return predicted_label
    except Exception as e:
        return f"Error: {e}"

# === Emotion to Emoji Mapping ===
emotion_emoji = {
    'angry': '😠',
    'calm': '😌',
    'disgust': '🤢',
    'fearful': '😨',
    'happy': '😄',
    'sad': '😢',
    'surprised': '😲',
    'neutral': '😐'
}

# === Music & Advice Mapping (All Public Videos) ===
suggestions = {
    'happy': {
        'music': "https://www.youtube.com/watch?v=ZbZSe6N_BXs",  # Pharrell - Happy
        'advice': "Keep spreading joy! You're awesome 😄"
    },
    'sad': {
        'music': "https://www.youtube.com/watch?v=J_ub7Etch2U",  # Sad Lofi
        'advice': "It's okay to feel sad. Take a deep breath and know brighter days are ahead 💙"
    },
    'angry': {
        'music': "https://www.youtube.com/watch?v=pXRviuL6vMY",  # Stressed Out
        'advice': "Try some calming breaths or take a short walk. You got this! 💪"
    },
    'calm': {
        'music': "https://www.youtube.com/watch?v=2OEL4P1Rz04",
        'advice': "Enjoy your calm state — it's a strength in itself. 🌿"
    },
    'fearful': {
        'music': "https://www.youtube.com/watch?v=F0U5JfGYx4c",
        'advice': "It's okay to be scared. You're stronger than you think. 🚀"
    },
    'disgust': {
        'music': "https://www.youtube.com/watch?v=qK1eFeVVyt0",  # Chill Vibe
        'advice': "Refocus on something that brings you joy. Refresh your vibe! 🎨"
    },
    'surprised': {
        'music': "https://www.youtube.com/watch?v=fLexgOxsZu0",  # Treasure
        'advice': "Surprises can be fun — embrace the unexpected! 🎉"
    },
    'neutral': {
        'music': "https://www.youtube.com/watch?v=hn3wJ1_1Zsg",  # Peaceful Music
        'advice': "You seem balanced — a perfect time to try something creative! 😊"
    }
}

# === Extract YouTube video ID ===
def extract_video_id(url):
    if "watch?v=" in url:
        return url.split("watch?v=")[-1]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1]
    return ""

# === UI ===
st.markdown("<h1 style='text-align: center;'>🎤 Emotion Recognition from Speech</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a <code>.wav</code> file and let the AI detect the emotion!</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("🎵 Upload your WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing emotion... 🎧"):
        emotion = predict_emotion("temp.wav", model, le)

    st.markdown("### 🧠 Predicted Emotion")
    if emotion in emotion_emoji:
        st.markdown(f"<div class='emotion-box'>{emotion.upper()} {emotion_emoji[emotion]}</div>", unsafe_allow_html=True)

        if emotion in suggestions:
            mood = suggestions[emotion]
            video_id = extract_video_id(mood['music'])

            st.markdown(f"""
                <div class="music-box">
                    <div class="music-title">🎶 Suggested Music</div>
                    <iframe width="100%" height="315" src="https://www.youtube.com/embed/{video_id}" 
                            frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
                            allowfullscreen></iframe>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### 💬 Advice for You")
            st.info(mood['advice'])

    else:
        st.warning(emotion)
