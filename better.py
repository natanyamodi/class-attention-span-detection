# from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
# import cv2

# # Load the face detection model
# face_model = YOLO("C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/face-detection-yolov8/yolov8n-face.pt")

# # Perform face detection
# face_results = face_model.predict(source="C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/faces.jpg", show=False)

# # Load image with OpenCV
# image = cv2.imread("C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/faces.jpg")

# # Process face detection results
# if isinstance(face_results, list) and len(face_results) > 0:
#     for result in face_results[0]:
#         x1, y1, x2, y2, conf, cls = result
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         # Draw bounding box around detected face
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

# # Display the image with bounding boxes
# cv2.imshow("Image with Bounding Boxes", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import cvzone
from ultralytics import YOLO
import tempfile
import os

# Initialize webcam
video_path = "C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/video.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLO model
facemodel = YOLO('C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/yolov8n-face.pt')

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 720))

    # Save frame as a temporary image file
    temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_frame.jpg')
    cv2.imwrite(temp_image_path, frame)

    # Predict on the single frame
    face_res = facemodel.predict(temp_image_path, conf=0.40)

    # Check if face_res is a list
    if isinstance(face_res, list):
        # If it's a list, assume it contains detections
        detections = face_res
    else:
        # If it's not a list, assume it's a YOLO result object and extract detections
        detections = face_res.xyxy[0]

    # Draw bounding boxes on the frame
    for detection in detections:
        for info in detection:
            x1, y1, x2, y2, conf, cls = info
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

    # Display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()







