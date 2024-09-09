# import cv2
# import cvzone
# from ultralytics import YOLO
# import tempfile
# import os
# from cvzone.FaceMeshModule import FaceMeshDetector
# import numpy as np
# from keras.models import model_from_json
# from scipy.spatial import distance as dist

# # Load emotion detection model
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Neutral", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# json_file = open('C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/Emotion_detection_with_CNN-main/model/emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)
# emotion_model.load_weights("C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/Emotion_detection_with_CNN-main/model/emotion_model.h5")
# print("Loaded emotion detection model")

# # Initialize webcam
# video_path = "C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/vid.mp4"
# cap = cv2.VideoCapture(video_path)

# # Load YOLO model for face detection
# facemodel = YOLO('C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/yolov8n-face.pt')

# # Initialize FaceMeshDetector for eye detection
# fm = FaceMeshDetector()
# rightEye = [463, 414, 286, 258, 257, 259, 268, 359, 255, 339, 254, 253, 252, 256, 341]
# leftEye = [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25]
# rightEye.extend(leftEye)

# # Set desired width and height for display
# display_width = 900
# display_height = 500

# def eye_aspect_ratio(eye):
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# # Define constants for EAR threshold and drowsiness detection
# EAR_THRESHOLD = 0.25
# CONSECUTIVE_FRAMES_DROWSY = 20

# # Variables for frame skipping
# skip_frames = 5
# frame_count = 0

# while cap.isOpened():
#     # Read frame from webcam
#     ret, frame = cap.read()
#     frame_count += 1

#     if not ret:
#         break

#     if frame_count % skip_frames != 0:
#         continue  # Skip processing this frame

#     frame = cv2.resize(frame, (1020, 720))

#     # Save frame as a temporary image file
#     temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_frame.jpg')
#     cv2.imwrite(temp_image_path, frame)

#     # Predict on the single frame
#     face_res = facemodel.predict(temp_image_path, conf=0.40)

#     # Check if face_res is a list
#     if isinstance(face_res, list):
#         # If it's a list, assume it contains detections
#         detections = face_res
#     else:
#         # If it's not a list, assume it's a YOLO result object and extract detections
#         detections = face_res.xyxy[0]

#     # Draw bounding boxes on the frame for faces
#     for detection in detections:
#         for info in detection:
#             x1, y1, x2, y2, conf, cls = info
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             h, w = y2 - y1, x2 - x1
#             cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

#             # Crop face region for emotion detection
#             roi_gray = frame[y1:y2, x1:x2]
#             roi_gray = cv2.resize(roi_gray, (48, 48))
#             roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
#             roi_gray = np.expand_dims(np.expand_dims(roi_gray, -1), 0)

#             # Predict emotion
#             emotion_prediction = emotion_model.predict(roi_gray)
#             maxindex = int(np.argmax(emotion_prediction))
#             emotion_label = emotion_dict[maxindex]

#             # Draw emotion label
#             cv2.putText(frame, emotion_label, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Perform face and eye detection on the frame
#     frame, faces = fm.findFaceMesh(frame, draw=0)

#     # Process detected faces and draw eye landmarks
#     if faces:
#         for face in faces:
#             for i in rightEye:
#                 points = face[i]
#                 cv2.circle(frame, points, 1, (0, 0, 255), 2)

#                 ear = eye_aspect_ratio([face[i] for i in rightEye])

#                 # Draw EAR value on the frame
#                 cv2.putText(frame, "EAR: {:.2f}".format(ear), (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                 # Check for drowsiness
#                 if ear <= EAR_THRESHOLD:
#                     frames_drowsy += 1
#                     if frames_drowsy >= CONSECUTIVE_FRAMES_DROWSY:
#                         # Perform action for drowsiness detection
#                         cv2.putText(frame, "Drowsy", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                 else:
#                     frames_drowsy = 0

#     # Display the frame with face and eye landmarks
#     cv2.imshow('Frame with Face and Eye Detection', frame)
    
#     if cv2.waitKey(50) & 0xFF == ord('q'):
#         break

# # Release the capture and close all windows
# cap.release()
# cv2.destroyAllWindows()

# Import specific functions or classes from inference.py


# import cv2
# import cvzone
# from ultralytics import YOLO
# import tempfile
# import os
# import numpy as np
# import mediapipe as mp

# # Load YOLO model for face detection
# facemodel = YOLO('C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/yolov8n-face.pt')

# # Initialize FaceMesh for head pose estimation
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# # Set desired width and height for display
# display_width = 900
# display_height = 500

# # Variables for frame skipping
# skip_frames = 5
# frame_count = 0

# # Initialize webcam
# video_path = "C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/video.mp4"
# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():
#     # Read frame from webcam
#     ret, frame = cap.read()
#     frame_count += 1

#     if not ret:
#         break

#     if frame_count % skip_frames != 0:
#         continue  # Skip processing this frame

#     frame = cv2.resize(frame, (1020, 720))

#     # Save frame as a temporary image file
#     temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_frame.jpg')
#     cv2.imwrite(temp_image_path, frame)

#     # Predict on the single frame using YOLO face detection
#     face_res = facemodel.predict(temp_image_path, conf=0.40)

#     # Check if face_res is a list
#     if isinstance(face_res, list):
#         # If it's a list, assume it contains detections
#         detections = face_res
#     else:
#         # If it's not a list, assume it's a YOLO result object and extract detections
#         detections = face_res.xyxy[0]

#     # Draw bounding boxes on the frame for faces
#     for detection in detections:
#         for info in detection:
#             x1, y1, x2, y2, conf, cls = info
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             h, w = y2 - y1, x2 - x1
#             cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

#             # Extract face region for head pose estimation
#             face_region = frame[y1:y2, x1:x2].copy()  # Make a copy of the face region

#             # Process face region for head pose estimation
#             frame_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(frame_rgb)

#             # Perform head pose estimation if face landmarks are detected
#             if results.multi_face_landmarks:
#                 for face_landmarks in results.multi_face_landmarks:
#                     face_3d = []
#                     face_2d = []
#                     for idx, lm in enumerate(face_landmarks.landmark):
#                         if idx==33 or idx == 263 or idx==1 or idx == 61 or idx==291 or idx==199:
#                             if idx==1:
#                                 nose_2d = (int(lm.x * face_region.shape[1]), int(lm.y * face_region.shape[0]))
#                                 node_3d = (lm.x * face_region.shape[1], lm.y * face_region.shape[0], lm.z * 3000)
#                             x, y = int(lm.x * face_region.shape[1]), int(lm.y * face_region.shape[0])
#                             face_2d.append([x, y])
#                             face_3d.append([x, y, lm.z])
#                     face_2d = np.array(face_2d, dtype=np.float64)
#                     face_3d = np.array(face_3d, dtype=np.float64)

#                     focal_length = 1 * face_region.shape[1]
#                     cam_matrix = np.array([[focal_length, 0, face_region.shape[0] / 2],
#                                            [0, focal_length, face_region.shape[1] / 2],
#                                            [0, 0, 1]])

#                     dist_matrix = np.zeros((4, 1), dtype=np.float64)
#                     success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

#                     rmat, jac = cv2.Rodrigues(rot_vec)

#                     angles, mtxR, ntxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
#                     x = angles[0] * 360
#                     y = angles[1] * 360
#                     z = angles[2] * 360

#                     if y <= -20:
#                         text = "looking left"
#                     elif y >= 20:
#                         text = "looking right"
#                     elif x <= -10:
#                         text = "looking down"
#                     elif x > 10:
#                         text = "looking up"
#                     else:
#                         text = "forward"

#                     cv2.putText(frame, text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Display the frame with face detection and head pose estimation
#     cv2.imshow('Frame with Face Detection and Head Pose Estimation', frame)
    
#     if cv2.waitKey(50) & 0xFF == ord('q'):
#         break

# # Release the capture and close all windows
# cap.release()
# cv2.destroyAllWindows()



import cv2
import cvzone
from ultralytics import YOLO
import tempfile
import os
import numpy as np
import mediapipe as mp
import math
import torch
import time 

right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]] # right eye landmark positions
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]] # left eye landmark positions
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]] # mouth landmark coordinates
states = ['alert', 'drowsy']
model_lstm_path = 'C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/Drowsiness-Detection-Mediapipe/models/clf_lstm_jit6.pth'
model = torch.jit.load(model_lstm_path)
model.eval()

def distance(p1, p2):
    ''' Calculate distance between two points
    :param p1: First Point 
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    '''
    return (((p1[:2] - p2[:2])**2).sum())**0.5

def eye_aspect_ratio(landmarks, eye):
    ''' Calculate the ratio of the eye length to eye width. 
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Eye aspect ratio value
    '''
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks):
    ''' Calculate the eye feature as the average of the eye aspect ratio for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Eye feature value
    '''
    return (eye_aspect_ratio(landmarks, left_eye) + \
    eye_aspect_ratio(landmarks, right_eye))/2

def mouth_feature(landmarks):
    ''' Calculate mouth feature as the ratio of the mouth length to mouth width
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Mouth feature value
    '''
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3)/(3*D)

def pupil_circularity(landmarks, eye):
    ''' Calculate pupil circularity feature.
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Pupil circularity for the eye coordinates
    '''
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
            distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
            distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
            distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
            distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
            distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
            distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
            distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4*math.pi*area)/(perimeter**2)

def pupil_feature(landmarks):
    ''' Calculate the pupil feature as the average of the pupil circularity for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Pupil feature value
    '''
    return (pupil_circularity(landmarks, left_eye) + \
        pupil_circularity(landmarks, right_eye))/2

def run_face_mp(image):
    ''' Get face landmarks using the FaceMesh MediaPipe model. 
    Calculate facial features using the landmarks.
    :param image: Image for which to get the face landmarks
    :return: Feature 1 (Eye), Feature 2 (Mouth), Feature 3 (Pupil), \
        Feature 4 (Combined eye and mouth feature), image with mesh drawings
    '''
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        # assume that only face is present in the image
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z]) # saving normalized landmark positions
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        # draw face mesh over image
        for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar/ear
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000

    return ear, mar, puc, moe, image

def calibrate(calib_frame_count=25):
    ''' Perform clibration. Get features for the neutral position.
    :param calib_frame_count: Image frames for which calibration is performed. Default Vale of 25.
    :return: Normalization Values for feature 1, Normalization Values for feature 2, \
        Normalization Values for feature 3, Normalization Values for feature 4
    '''
    ears = []
    mars = []
    pucs = []
    moes = []

    vid = "C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/vid.mp4"
    cap = cv2.VideoCapture(vid)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        ear, mar,puc, moe, image = run_face_mp(image)
        if ear != -1000:
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)

        cv2.putText(image, "Calibration", (int(0.02*image.shape[1]), int(0.14*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        if len(ears) >= calib_frame_count:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    ears = np.array(ears)
    mars = np.array(mars)
    pucs = np.array(pucs)
    moes = np.array(moes)
    return [ears.mean(), ears.std()], [mars.mean(), mars.std()], \
        [pucs.mean(), pucs.std()], [moes.mean(), moes.std()]

def get_classification(input_data):
    ''' Perform classification over the facial  features.
    :param input_data: List of facial features for 20 frames
    :return: Alert / Drowsy state prediction
    '''
    model_input = []
    model_input.append(input_data[:5])
    model_input.append(input_data[3:8])
    model_input.append(input_data[6:11])
    model_input.append(input_data[9:14])
    model_input.append(input_data[12:17])
    model_input.append(input_data[15:])
    model_input = torch.FloatTensor(np.array(model_input))
    preds = torch.sigmoid(model(model_input)).gt(0.5).int().data.numpy()
    return int(preds.sum() >= 2)
# Load YOLO model for face detection
facemodel = YOLO('C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/yolov8n-face.pt')

# Initialize FaceMesh for facial feature extraction
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Set desired width and height for display
display_width = 900
display_height = 500

# Variables for frame skipping
skip_frames = 15
frame_count = 0

# Initialize webcam
video_path = "C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/video.mp4"
cap = cv2.VideoCapture(0)


# Define normalization values for facial features
ears_norm = [0, 1]   # Example normalization values, replace with your actual values
mars_norm = [0, 1]
pucs_norm = [0, 1]
moes_norm = [0, 1]
label = None
input_data = []

left_right_counter = 0
drowsiness_count = 0
while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        break

    # frame_count += 1
    # if frame_count % skip_frames != 0:
    #     continue

    frame = cv2.resize(frame, (1020, 720))

    # Save frame as a temporary image file
    temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_frame.jpg')
    cv2.imwrite(temp_image_path, frame)

    # Predict on the single frame using YOLO face detection
    face_res = facemodel.predict(temp_image_path, conf=0.40)

    # Check if face_res is a list
    if isinstance(face_res, list):
        # If it's a list, assume it contains detections
        detections = face_res
    else:
        # If it's not a list, assume it's a YOLO result object and extract detections
        detections = face_res.xyxy[0]

    
    # Process each detected face
    for detection in detections:
        total_ear = 0
        total_puc = 0
        total_mar = 0
        total_moe = 0
        face_count = 0
        for info in detection:
            x1, y1, x2, y2, conf, cls = info
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

            # Extract face region for facial feature extraction
            face_region = frame[y1:y2, x1:x2].copy()  # Make a copy of the face region
            ear, mar, puc, moe, _ = run_face_mp(face_region)
            if ear != -1000:
                total_ear += ear
                total_mar += mar
                total_puc += puc
                total_moe += moe
                face_count += 1

            if len(input_data) == 20:
                input_data.pop(0)
            input_data.append([total_ear, total_ear, total_puc, total_moe])

            if len(input_data) == 20:
                label = get_classification(input_data)
            
            
            # Draw bounding box and text for each detection
            if face_count > 0:
                avg_ear = total_ear / face_count
                avg_mar = total_mar / face_count
                avg_puc = total_puc / face_count
                avg_moe = total_moe / face_count

                # Display the average values
                cv2.putText(frame, "Average EAR: %.2f" % avg_ear, (int(0.02 * frame.shape[1]), int(0.07 * frame.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, "Average MAR: %.2f" % avg_mar, (int(0.27 * frame.shape[1]), int(0.07 * frame.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, "Average PUC: %.2f" % avg_puc, (int(0.52 * frame.shape[1]), int(0.07 * frame.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, "Average MOE: %.2f" % avg_moe, (int(0.77 * frame.shape[1]), int(0.07 * frame.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                if label is not None:
                    if label == 0:
                        label_text = "Alert"
                        color = (0, 255, 0)  # Green for "Alert"
                    else:
                        label_text = "Drowsy"
                        color = (0, 0, 255)  # Red for "Drowsy"
                        drowsiness_count += 1
                    cv2.putText(frame, label_text, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        
        


            # Process face region for facial feature extraction
            frame_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            # Perform facial feature extraction if face landmarks are detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_3d = []
                    face_2d = []
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx==33 or idx == 263 or idx==1 or idx == 61 or idx==291 or idx==199:
                            if idx==1:
                                nose_2d = (int(lm.x * face_region.shape[1]), int(lm.y * face_region.shape[0]))
                                node_3d = (lm.x * face_region.shape[1], lm.y * face_region.shape[0], lm.z * 3000)
                            x, y = int(lm.x * face_region.shape[1]), int(lm.y * face_region.shape[0])
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * face_region.shape[1]
                    cam_matrix = np.array([[focal_length, 0, face_region.shape[0] / 2],
                                        [0, focal_length, face_region.shape[1] / 2],
                                        [0, 0, 1]])

                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    rmat, jac = cv2.Rodrigues(rot_vec)

                    angles, mtxR, ntxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    if y <= -40:
                        text = "looking left"
                        left_right_counter += 1
                    elif y >= 40:
                        text = "looking right"
                        left_right_counter += 1
                    elif x <= -10:
                        text = "looking down"
                    elif x > 10:
                        text = "looking up"
                    else:
                        text = "forward"

                    cv2.putText(frame, text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    ear_main = 0

    cv2.imshow('Frame with Face Detection and Facial Feature Extraction', frame)
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break


total_frames = frame_count
attention_span_percentage = ((total_frames - (left_right_counter + drowsiness_count)) / 70) * 100
print("Class attention span percentage:", attention_span_percentage)


# Release the capture and close all windows
cap.release()

try:
    # Code for calibration and main application
    print ('Starting calibration. Please be in neutral state')
    time.sleep(1)
    ears_norm, mars_norm, pucs_norm, moes_norm = calibrate()

    video_path = 'C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/video.mp4'

    print('Starting main application on video...')
    time.sleep(1)
    

except KeyboardInterrupt:
    print("Keyboard interrupt detected. Closing the file...")
    face_mesh.close()  # Close any open files or resources

