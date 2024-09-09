import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh =  mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.8, min_tracking_confidence=0.8)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture("C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/vid.mp4")

display_width = 900
display_height = 500

while cap.isOpened():
    success, image = cap.read()
    start = time.time()
    image = cv2.resize(image, (display_width, display_height))  # Resizing the image

    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx==33 or idx == 263 or idx==1 or idx == 61 or idx==291 or idx==199:
                    if idx==1:
                        nose_2d = (int(lm.x * display_width), int(lm.y * display_height))
                        node_3d = (lm.x * display_width, lm.y * display_height, lm.z * 3000)
                    x, y = int(lm.x * display_width), int(lm.y * display_height)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * display_width
            cam_matrix = np.array([[focal_length, 0, display_height / 2],
                                   [0, focal_length, display_width / 2],
                                   [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)

            angles, mtxR, ntxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y <= -10:
                text = "looking left"
            elif y >= 10:
                text = "looking right"
            elif x <= -10:
                text = "looking down"
            elif x > 10:
                text = "looking up"
            else:
                text = "forward"



            # nose_3d_projection, jacobian = cv2.projectPoints(node_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            # p1 = (int(nose_2d[0]), int(nose_2d[1]))
            # p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            # cv2.line(image, p1, p2, (255, 0, 0), 3)

            cv2.putText(image, text, (int(nose_2d[0]), int(nose_2d[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=drawing_spec,
            #     connection_drawing_spec=drawing_spec
            # )

    cv2.imshow('head pose estimation', image)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()


