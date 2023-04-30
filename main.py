import cv2
import mediapipe as mp

# MediaPipe hands module initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Connection between landmarks
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

# Drawing specs for connections and landmarks
connection_drawing_spec = {connection: mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2) for connection in connections}
landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

# OpenCV video capture initialization
cap = cv2.VideoCapture(0)

# Map hand gesture to text
gesture_to_text = {
    "rock": "Rock",
    "paper": "Paper",
    "scissor": "Scissor",
}

while cap.isOpened():
    # Read the video frame
    success, image = cap.read()

    if not success:
        print("Failed to read a frame from the camera")
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect hands in the image
    results = hands.process(image_rgb)

    # Check if hands were detected
    if results.multi_hand_landmarks:
        # Iterate through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmarks as a list
            landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark]
            
            # Check if the hand gesture corresponds to "rock", "paper", or "scissor"
            if landmarks[8][1] < landmarks[6][1] < landmarks[10][1]:
                text = gesture_to_text["rock"]
            elif landmarks[6][2] < landmarks[8][2] and landmarks[10][2] < landmarks[8][2]:
                text = gesture_to_text["paper"]
            elif landmarks[6][2] > landmarks[8][2] and landmarks[10][2] > landmarks[8][2]:
                text = gesture_to_text["scissor"]
            else:
                text = ""
            
            # Draw the hand landmarks on the image
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec,
                connection_drawing_spec,
            )
            
            # Draw the text on the image
            if text:
                cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (cv2.imshow("Rock, Paper, Scissors", image)))
        
        # Check for 'q' key pressed to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # If hands are not detected, display a message on the screen
        cv2.putText(
            image,
            "No hands detected",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Rock, Paper, Scissors", image)
        
        # Check for 'q' key pressed to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()