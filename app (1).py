from flask import Flask, render_template, Response, request, jsonify
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from gtts import gTTS
import pygame
import io
import time

app = Flask(__name__)

# Initialize pygame
pygame.mixer.init()

# Initialize camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")

offset = 20
imgSize = 300
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

# Word storage
recognized_word = ""
current_letter = ""


def generate_frames():
    global current_letter
    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            h_img, w_img, _ = img.shape
            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = min(w_img, x + w + offset), min(h_img, y + h + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                aspectRatio = h / w

                try:
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    current_letter = labels[index]

                    cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                                (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, current_letter, (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x-offset, y-offset),
                                (x + w+offset, y + h+offset), (255, 0, 255), 4)

                except Exception as e:
                    print("Error processing hand:", e)
        else:
            current_letter = ""  # ✅ Clear current letter when no hand is present

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/action', methods=['POST'])
def action():
    global recognized_word, current_letter
    action_type = request.form.get('action')

    if action_type == 'add':
        # Only add if a letter is currently detected
        if current_letter != "":
            recognized_word += current_letter
    elif action_type == 'clear':
        recognized_word = recognized_word[:-1] if recognized_word else ""
    elif action_type == 'space':
        recognized_word += " "
    elif action_type == 'speak':
        if recognized_word:
            tts = gTTS(text=recognized_word, lang='en')
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            pygame.mixer.music.load(audio_buffer, 'mp3')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
    return '', 204


@app.route('/shutdown', methods=['POST'])
def shutdown():
    global cap
    cap.release()
    cv2.destroyAllWindows()
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

@app.route('/word')
def get_word():
    global recognized_word
    return jsonify({'word': recognized_word})

if __name__ == '__main__':
    app.run(debug=True)