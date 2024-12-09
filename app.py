import cv2
import mediapipe as mp
from flask import Flask, render_template, Response
import numpy as np

# Initialize MediaPipe Pose and Hand models
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Flask application setup
app = Flask(__name__)

# Variables to track waving gesture
previous_x = None
waving_status = "Not Waving"

# Function to detect poses, add text overlay, and count fingers
def detect_pose_and_add_text(frame):
    global previous_x, waving_status
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize status variables
    status = "No Pose Detected"
    hand_status = "No Hands Detected"
    finger_count = 0

    # Pose Detection
    pose_result = pose.process(frame_rgb)
    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        status = "Human Detected"

    # Hand Detection
    hand_result = hands.process(frame_rgb)
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count raised fingers
            finger_tips = [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP
            ]
            finger_joints = [
                mp_hands.HandLandmark.THUMB_IP,
                mp_hands.HandLandmark.INDEX_FINGER_PIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                mp_hands.HandLandmark.RING_FINGER_PIP,
                mp_hands.HandLandmark.PINKY_PIP
            ]
            for tip, joint in zip(finger_tips, finger_joints):
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[joint].y:
                    finger_count += 1
            hand_status = "Hands Detected"

            # Waving detection
            current_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            if previous_x is not None:
                if abs(current_x - previous_x) > 0.03:  # Threshold for waving motion
                    waving_status = "Waving Detected"
                else:
                    waving_status = "Not Waving"
            previous_x = current_x

    # Overlay text on the video feed
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Hand Status: {hand_status} - Fingers: {finger_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Waving: {waving_status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Pranjal Bhatt", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Flask route for the homepage
@app.route('/')
def index():
    return render_template('index.html')  # Frontend template

# Function to generate frames for the video stream
def gen_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Apply pose detection and add overlay text
        frame = detect_pose_and_add_text(frame)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of the video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

# Route to stream video
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

