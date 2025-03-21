from flask import Flask, render_template, Response, request
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
from googletrans import Translator
import threading

app = Flask(__name__)

# Initialize hand detector, classifier, and translator
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")
cap = cv2.VideoCapture(0)
translator = Translator()

# Define labels for classification
labels = [
    "Are you ready?", "Excuse me", "Exercise", "Fine", "Finish", "Good",
    "Hello", "Help", "How are you", "I Love You", "Me", "morning", "No",
    "See you later", "Sorry", "Stop", "Thank you", "Where", "wrong", "Yes", "you"
]

# Default language for translation
language = 'en'  # Default to English
frame_lock = threading.Lock()  # To manage camera frame updates

@app.route('/', methods=['GET', 'POST'])
def index():
    global language
    if request.method == 'POST':
        language = request.form['language']
    return render_template('index.html', language=language)

def generate_frames():
    global cap
    while True:
        with frame_lock:  # Ensuring only one frame capture at a time
            success, img = cap.read()
            if not success:
                break

            imgOutput = img.copy()
            hands, img = detector.findHands(img)

            subtitle_text = ""  # Default empty subtitle

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((300, 300, 3), np.uint8) * 255
                imgCrop = img[y-20:y + h+20, x-20:x + w+20]
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = 300 / h
                    wCal = int(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, 300))
                    wGap = int((300 - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                else:
                    k = 300 / w
                    hCal = int(k * h)
                    imgResize = cv2.resize(imgCrop, (300, hCal))
                    hGap = int((300 - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Set the subtitle text to the detected label
                subtitle_text = labels[index]

                # Translate the subtitle text if the selected language is not English
                if language != 'en':
                    translated = translator.translate(subtitle_text, src='en', dest=language)
                    subtitle_text = translated.text

            # Overlay subtitle text at the bottom of the frame
            height, width, _ = imgOutput.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size = cv2.getTextSize(subtitle_text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 30  # Position above the bottom edge

            if subtitle_text:
                cv2.putText(imgOutput, subtitle_text, (text_x, text_y), font, font_scale, (255,255, 255), thickness, cv2.LINE_AA)

            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', imgOutput)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def camera_thread():
    """ Threaded function to run the camera feed """
    cap.open(0)  # Ensure the camera is open
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    # Start the camera in a separate thread
    thread = threading.Thread(target=camera_thread)
    thread.start()
