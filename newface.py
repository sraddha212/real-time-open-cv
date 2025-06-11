from mtcnn import MTCNN
import cv2
detector = MTCNN()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.detect_faces(frame)

    for result in results:
        x, y, width, height = result['box']
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 0, 0), 2)
        for key, point in result['keypoints'].items():
            cv2.circle(frame, point, 2, (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
