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

# Track the previous set of landmarks to detect movement
previous_landmarks = None
status_history = []  # History of statuses for smoothing

# Activity debounce to avoid fluctuations
debounce_counter = {
    "Running": 0,
    "Still": 0,
    "Jumping": 0,
    "Punching": 0,
    "Waving": 0,
}

# Threshold for activity recognition to avoid over-sensitivity
MOVEMENT_THRESHOLD = 0.05  # Reduce sensitivity, adjust as necessary

# Function to count raised fingers based on landmarks
def count_raised_fingers(hand_landmarks):
    # Finger tip and joint landmarks based on Mediapipe documentation
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
    
    count = 0
    for tip, joint in zip(finger_tips, finger_joints):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[joint].y:
            count += 1  # Finger is raised if tip is above the joint

    return count

# Function to detect poses, add text overlay based on motion, detect actions, and count fingers
def detect_pose_and_add_text(frame):
    global previous_landmarks
    global debounce_counter

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Default value for status
    status = "No Pose Detected"  # Set default status value for cases where no pose is detected
    hand_status = "No Hand Detected"  # Default hand status value
    finger_count = 0  # Default finger count

    # Pose Detection
    pose_result = pose.process(frame_rgb)
    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Convert pose landmarks to a numpy array for easier processing
        current_landmarks = pose_result.pose_landmarks.landmark

        # Calculate movement by comparing current and previous landmarks
        if previous_landmarks is not None:
            # Calculate the Euclidean distance between current and previous landmarks
            movement = np.linalg.norm(
                np.array([(lm.x, lm.y, lm.z) for lm in current_landmarks]) - 
                np.array([(lm.x, lm.y, lm.z) for lm in previous_landmarks]), axis=1).mean()
            
            # Determine status based on movement
            if movement > MOVEMENT_THRESHOLD:
                status = "Running"
                debounce_counter["Running"] += 1
                if debounce_counter["Running"] > 5:  # Hold status for 5 frames to prevent flickering
                    status = "Running"
                    debounce_counter["Still"] = 0
                    debounce_counter["Jumping"] = 0
                    debounce_counter["Punching"] = 0
            else:
                status = "Still"
                debounce_counter["Still"] += 1
                if debounce_counter["Still"] > 5:
                    status = "Still"
                    debounce_counter["Running"] = 0
                    debounce_counter["Jumping"] = 0
                    debounce_counter["Punching"] = 0
            
            # Detect Jumping based on Y-axis position of knees (UPWARD movement)
            left_knee = current_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = current_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            if left_knee.y < previous_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y and \
               right_knee.y < previous_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y:
                status = "Jumping"
                debounce_counter["Jumping"] += 1
                if debounce_counter["Jumping"] > 5:
                    status = "Jumping"
                    debounce_counter["Running"] = 0
                    debounce_counter["Still"] = 0
                    debounce_counter["Punching"] = 0

            # Detect Punching based on elbow positions
            left_elbow = current_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = current_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            if left_elbow.y < previous_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y and \
               right_elbow.y < previous_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y:
                status = "Punching"
                debounce_counter["Punching"] += 1
                if debounce_counter["Punching"] > 5:
                    status = "Punching"
                    debounce_counter["Running"] = 0
                    debounce_counter["Still"] = 0
                    debounce_counter["Jumping"] = 0

        # Update previous landmarks for the next frame
        previous_landmarks = current_landmarks
    else:
        status = "No Pose Detected"  # Default status when no pose is detected

    # Hand Detection (Optional, you can keep this for waving hands and finger counting)
    hand_result = hands.process(frame_rgb)
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = count_raised_fingers(hand_landmarks)  # Count raised fingers
        hand_status = "Waving Hand"
    
    # Store the status for smoothing
    status_history.append(status)
    if len(status_history) > 10:  # Smooth over the last 10 frames
        status_history.pop(0)
    
    smoothed_status = max(set(status_history), key=status_history.count)

    # Overlay the smoothed status text and finger count
    cv2.putText(frame, f"Status: {smoothed_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Hand Status: {hand_status} - Fingers: {finger_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Overlay your name on the top-right corner
    cv2.putText(frame, "pranjal bhatt", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

# Flask route for video feed
@app.route('/')
def index():
    return render_template('index.html')  # Frontend template

def gen_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = cap.read()
        if not success:
            print("Error reading frame, skipping...")
            continue  # Skip frame if reading fails
        # Apply pose detection and add overlay text
        frame = detect_pose_and_add_text(frame)
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue  # Skip this frame if encoding fails
        frame = buffer.tobytes()

        # Stream the frame to the webpage
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    # Release the camera when done
    cap.release()
# Route to stream video
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
