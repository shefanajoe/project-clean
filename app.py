import cv2
from flask import Flask, render_template, Response, jsonify, request
import tensorflow as tf
import numpy as np
import random

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained emotion detection model (emotion_model11.h5)
emotion_model = tf.keras.models.load_model('emotion_model11.h5')

# Load OpenCV's Haarcascade Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels for FER-2013 dataset (7 emotions)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Global variable to store the detected emotion
current_emotion = "Detecting..."

def detect_emotion(frame):
    global current_emotion

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        emotion_probs = emotion_model.predict(face)
        max_index = np.argmax(emotion_probs)
        current_emotion = emotion_labels[max_index]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_emotion(frame)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/text')
def text_input():
    return render_template('index.html')

@app.route('/video')
def video_input():
    return render_template('video_input.html')  # You can create this file later

@app.route('/lazy')
def lazy_recommendation():
    random_emotion = random.choice(emotion_labels)
    emotion_links = get_music_link(random_emotion)
    return render_template('lazy.html', emotion=random_emotion, links=emotion_links)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_emotion')
def detect_emotion_api():
    emotion_links = get_music_link(current_emotion)
    return jsonify({
        "emotion": current_emotion,
        "links": {
            "Spotify": emotion_links["Spotify"],
            "YouTube": emotion_links["YouTube"],
            "Apple Music": emotion_links["Apple Music"]
        }
    })

@app.route('/get_links', methods=['POST'])
def get_links():
    data = request.get_json()
    mood = data.get('mood', 'Happy')
    links = get_music_link(mood)
    return jsonify(links)

def get_music_link(emotion):
    emotion_to_playlist = {
        "Happy": {
            "Spotify": "https://open.spotify.com/playlist/37i9dQZF1DWZKuerrwoAGz?si=LN905NWRRtqvd4nzBv5rBw",
            "YouTube": "https://www.youtube.com/channel/UCtBBEQw4SFYOcrZKzY7quCQ",
            "Apple Music": "https://music.apple.com/us/playlist/happy-playlist/pl.8b3e446f032344e29b3d9ff1ab4c685f"
        },
        "Sad": {
            "Spotify": "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1?si=Mi82heteQEeStjaVT-MpsQ&pi=3zd0K3NSRVWST",
            "YouTube": "https://www.youtube.com/watch?app=desktop&v=s7FTAxw37hk&t=251s",
            "Apple Music": "https://music.apple.com/us/playlist/sad-playlist/pl.9c87f12345ff"
        },
        "Angry": {
            "Spotify": "https://open.spotify.com/playlist/37i9dQZF1EIhuCNl2WSFYd?si=zQO5AeOlRfaj2rlZJLPUlw&pi=lgKHsHENTUSSR&nd=1&dlsi=e9f3f377667b4e15",
            "YouTube": "https://www.youtube.com/watch?v=3Z0_rDmKlc0&pp=ygUMI2FzaGl0b3ZoaW5h",
            "Apple Music": "https://music.apple.com/us/playlist/angry-playlist/pl.2babc13456df"
        },
        "Fear": {
            "Spotify": "https://open.spotify.com/playlist/6vEYfoIeIIzC87k94JDcYj?si=iUnevDmmRl-aCHSWxDODFg&pi=bI1z1tAWS6yLX",
            "YouTube": "https://www.youtube.com/watch?v=CKpbdCciELk",
            "Apple Music": "https://music.apple.com/us/playlist/fear-playlist/pl.abcdef1234"
        },
        "Surprise": {
            "Spotify": "https://open.spotify.com/show/2EJPkN70Jq6UaJnwaZa7tJ?si=TDB_bCoITTWgx6FTQKLMYQ&nd=1&dlsi=7f965ee080f94970",
            "YouTube": "https://www.youtube.com/watch?v=4IOnFtphS6g",
            "Apple Music": "https://music.apple.com/us/playlist/surprise-playlist/pl.5678ijkl9012"
        },
        "Neutral": {
            "Spotify": "https://open.spotify.com/playlist/30VCumZ3LKfQQqD3alyltu?si=2Z5xxNRdTKON7Q8V8ElVyg&pi=uZBLDJ43RpScq&nd=1&dlsi=fc2ae31022a34a9b",
            "YouTube": "https://www.youtube.com/watch?v=WpZ1WiqCf94",
            "Apple Music": "https://music.apple.com/us/playlist/neutral-playlist/pl.1234mnop"
        },
        "Disgust": {
            "Spotify": "https://open.spotify.com/playlist/4y61zEEdKhJMaKAGO503Wx?si=TAiTv1J9RoOqM_qWnfV6hA&pi=9YOGEB6DQ5aSb&nd=1&dlsi=b6bfd3d073934516",
            "YouTube": "https://www.youtube.com/watch?v=R1Jve1AeQ9Y",
            "Apple Music": "https://music.apple.com/us/playlist/disgust-playlist/pl.abcd12345"
        }
    }
    return emotion_to_playlist.get(emotion, {
        "Spotify": "#",
        "YouTube": "#",
        "Apple Music": "#"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
