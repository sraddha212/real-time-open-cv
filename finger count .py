import cv2
import mediapipe as mp

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def count_fingers(hand_landmarks, hand_label):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    count = 0

    if hand_label == "Right":
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
            count += 1
    else:
        if hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_tip - 1].x:
            count += 1

    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1

    return count

cap = cv2.VideoCapture(0)

cv2.namedWindow('Finger Counting', cv2.WINDOW_NORMAL)

fullscreen = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    total_fingers = 0
    hand_counts = []

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = count_fingers(hand_landmarks, hand_label)
            total_fingers += fingers
            hand_counts.append((hand_label, fingers))

            # Draw circle on  finger tips
            for tip in [4, 8, 12, 16, 20]:
                x = int(hand_landmarks.landmark[tip].x * frame.shape[1])
                y = int(hand_landmarks.landmark[tip].y * frame.shape[0])
                cv2.circle(frame, (x, y), 12, (0, 255, 255), cv2.FILLED)  
                cv2.circle(frame, (x, y), 14, (255, 255, 255), 2)         
    y0 = 30
    dy = 40
    for i, (label, count) in enumerate(hand_counts):
        # hand counts  green color
        cv2.putText(frame, f'{label} Hand: {count} fingers', (10, y0 + i * dy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if hand_counts:
        # Total fingers 
        cv2.putText(frame, f'Total Fingers: {total_fingers}', (10, y0 + len(hand_counts) * dy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Finger Counting', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty('Finger Counting', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty('Finger Counting', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()
