# human actvity recognition project(1)(2024)
 it includes 5-6 human activity with accuracy of 70-80%

 Project Title
Real-Time Pose, Gesture, and Finger Detection System Using OpenCV, MediaPipe, and Flask

Project Overview
This project is a real-time video-based activity recognition system that leverages MediaPipe for body pose and hand landmark detection to identify various human activities, gestures, and finger counts. The application captures live video from a webcam, processes each frame to analyze body and hand movements, and streams the annotated video feed to a web interface using Flask. It detects activities such as running, jumping, punching, and waving and can recognize specific hand gestures by counting the number of raised fingers.

Key Components and Functionality
1. Pose Detection with MediaPipe Pose Model
The MediaPipe Pose model identifies key body landmarks, such as the elbows, knees, shoulders, and wrists, which are essential in determining human poses.
By tracking these landmarks, the system recognizes specific physical activities like:
Running: Detected by assessing the speed and movement pattern of the body landmarks.
Jumping: Detected by monitoring the vertical displacement of the knees.
Punching: Detected by tracking the forward motion of elbows.
Stillness: Identified when there is minimal overall body movement.
To ensure accuracy, the system implements a debouncing technique that maintains activity status only if the detected movement persists across multiple frames. This method prevents fluctuations, so short-term or minor movements don’t mistakenly alter the recognized activity.
2. Hand Detection and Finger Counting with MediaPipe Hands Model
The MediaPipe Hands model detects the hand’s position and landmarks on individual fingers.
This enables the system to:
Recognize if a hand is waving.
Count the number of fingers raised, with the potential to differentiate between gestures (e.g., peace sign, thumbs up).
When a hand is detected, the system counts the fingers by analyzing the relative positions of hand landmarks. This adds an extra layer of interactivity, allowing the user to see their finger count displayed on the screen in real time.
3. Real-Time Video Processing and Text Overlay
OpenCV handles the video capture and frame-by-frame processing.
Each frame is converted to RGB, processed through the MediaPipe models, and annotated with various overlays:
Activity Status: Displays the current detected activity, such as "Running," "Jumping," "Still," or "Punching."
Hand Status and Finger Count: Shows if a hand is detected (e.g., "Waving Hand") and the count of raised fingers.
Personal Labeling: Adds the user’s name (or another identifier) to personalize the display.
The project incorporates smoothing logic, where the final displayed status is based on the most common status in the last few frames. This reduces rapid fluctuations and provides a stable user experience.
4. Web Interface with Flask for Video Streaming
A Flask web application serves the processed video feed to users in real time.
The application includes:
Webpage Rendering: The homepage displays the live video feed.
Video Feed Endpoint: A dedicated route (/video) streams the continuously updated frames to the webpage using the MJPEG format, enabling low-latency viewing.
5. Smoothing and Debouncing for Enhanced Accuracy
The system smooths activity detection by holding the detected status only after the same activity is recognized across multiple frames, reducing flickering or instability in the displayed output.
This debouncing technique helps ensure activities like running or jumping are consistently detected without noise from small, unintentional movements.
Workflow Summary
Video Capture: OpenCV initializes the webcam, capturing the video in real-time.
Pose and Hand Processing:
Each video frame is converted from BGR to RGB, as required by MediaPipe.
Pose landmarks are processed to detect movements and activities, while hand landmarks are used for gesture recognition and finger counting.
Activity and Gesture Recognition:
Specific logic evaluates pose and hand landmarks to classify actions (running, jumping, punching) and hand gestures.
Landmarks are compared across frames to detect consistent movements.
Text Overlay:
Activity, hand status, finger count, and a user label are overlaid on the video frame.
Streaming to Webpage: The annotated frame is encoded and sent as an MJPEG stream to a webpage via Flask.
Applications
Fitness and Exercise Tracking: This system can be adapted to monitor exercise forms and movements, providing feedback on various activities and potentially tracking workout progress.
Interactive Gesture-Based Controls: The hand tracking and finger counting components could be further developed for use in gesture-controlled applications or devices.
Virtual Reality (VR) and Gaming: With real-time pose and gesture recognition, the project has potential applications in VR systems and gesture-based gaming environments.
Educational Tools: This system could be adapted to create engaging educational tools for learning about human anatomy or for physical activity tracking in schools.
Technical Stack
Programming Language: Python
Computer Vision: OpenCV
Machine Learning Model: MediaPipe Pose and Hands
Web Framework: Flask
Summary of Benefits and Potential Enhancements
The project demonstrates the integration of real-time video processing, machine learning-based landmark detection, and web streaming, making it a powerful and interactive tool. It provides instant visual feedback on user movements, can recognize hand gestures, and is suitable for applications in fitness, gesture-based control, and immersive experiences.

Potential Enhancements could include:

Expanding gesture recognition to detect more complex hand movements.
Adding additional activity types, such as squats or push-ups, for a broader fitness application.
Integrating the system with cloud-based databases to store activity data and analyze user trends over time.
This comprehensive setup makes your project both a sophisticated and versatile tool, merging real-time computer vision with practical applications in fitness, VR, education, and interactive control.







