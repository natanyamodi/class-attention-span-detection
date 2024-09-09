import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh

faceDetection = mpFaceDetection.FaceDetection(0.8)
faceMesh = mpFaceMesh.FaceMesh()

# Set the desired display size
display_width = 900  # Adjust this value as needed
display_height = 500  # Adjust this value as needed

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face detection
    results_face = faceDetection.process(imgRGB)

    if results_face.detections:
        for id, detection in enumerate(results_face.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)

            # Get landmarks for face mesh
            landmarks = faceMesh.process(imgRGB).multi_face_landmarks
            if landmarks:
                for face_id, landmark in enumerate(landmarks):
                    # Left eye
                    eye_left = (int(landmark.landmark[159].x * iw), int(landmark.landmark[159].y * ih))
                    # Right eye
                    eye_right = (int(landmark.landmark[386].x * iw), int(landmark.landmark[386].y * ih))

                    # Draw green boxes around eyes
                    cv2.rectangle(img, (eye_left[0] - 20, eye_left[1] - 20), (eye_left[0] + 20, eye_left[1] + 20), (0, 255, 0), 2)
                    cv2.rectangle(img, (eye_right[0] - 20, eye_right[1] - 20), (eye_right[0] + 20, eye_right[1] + 20), (0, 255, 0), 2)

                    # Mouth landmarks
                    mouth_left = (int(landmark.landmark[61].x * iw), int(landmark.landmark[61].y * ih))
                    mouth_right = (int(landmark.landmark[91].x * iw), int(landmark.landmark[91].y * ih))
                    mouth_top = (int(landmark.landmark[13].x * iw), int(landmark.landmark[13].y * ih))
                    mouth_bottom = (int(landmark.landmark[14].x * iw), int(landmark.landmark[14].y * ih))

                    # Calculate mouth width and height
                    mouth_width = mouth_right[0] - mouth_left[0]
                    mouth_height = mouth_bottom[1] - mouth_top[1]

                    # Calculate mouth aspect ratio
                    mouth_aspect_ratio = (mouth_width / mouth_height) if mouth_height > 0 else 0

                    # Emotion detection
                    if mouth_aspect_ratio > 0.6:
                        emotion = "Happy"
                    elif mouth_aspect_ratio < 0.4:
                        emotion = "Sad"
                    else:
                        emotion = "Neutral"

                    # Display emotion beside detection score
                    cv2.putText(img, f'Emotion {face_id + 1}: {emotion}', (40, 120 + face_id * 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)

    # Resize the video frame to fit the desired display size
    img_resized = cv2.resize(img, (display_width, display_height))

    cv2.imshow("Image", img_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
