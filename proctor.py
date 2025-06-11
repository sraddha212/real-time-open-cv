import cv2

# Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    status = "Status: OK"
    color = (0, 255, 0)  # Green

    # chech faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        status = "Error: No face detected!"
        color = (0, 0, 255)
    elif len(faces) > 1:
        status = f"Error: {len(faces)} faces detected!"
        color = (0, 0, 255)
    else:
        # Only one face check... now check for eyes
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) < 2:
            status = "Error: Eyes not clearly visible!"
            color = (0, 0, 255)

        #face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

    # status
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Proctoring', frame)

    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
