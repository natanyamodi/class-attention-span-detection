# import cv2 as cv
# from cvzone.FaceMeshModule import FaceMeshDetector

# # Initialize FaceMeshDetector for eye detection
# fm = FaceMeshDetector()
# rightEye = [463, 414, 286, 258, 257, 259, 268, 359, 255, 339, 254, 253, 252, 256, 341]
# leftEye = [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25]
# rightEye.extend(leftEye)

# # Choose between video file or webcam feed
# video_path = "C:/Users/Natanya Modi/OneDrive/Desktop/Projects/attention-span-detection/video.mp4"  

# # Open the video capture
# cap = cv.VideoCapture(0)

# # Set desired width and height for display
# display_width = 900
# display_height = 500

# while cap.isOpened():
#     # Read a frame from the video
#     ret, frame = cap.read()

#     # Break the loop if no frame is retrieved
#     if not ret:
#         break

#     # Resize the frame to desired width and height
#     frame = cv.resize(frame, (display_width, display_height))

#     # Perform face and eye detection on the frame
#     frame, faces = fm.findFaceMesh(frame, draw=0)

#     # Process detected faces and draw eye landmarks
#     if faces:
#         for face in faces:
#             for i in rightEye:
#                 points = face[i]
#                 cv.circle(frame, points, 1, (0, 0, 255), 2)

#     # Display the frame with eye landmarks
#     cv.imshow('Frame with Eye Detection', frame)

#     # Exit the loop if 'q' is pressed
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all windows
# cap.release()
# cv.destroyAllWindows()




