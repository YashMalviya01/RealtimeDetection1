import os
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time
import numpy as np

# Define the folder where images will be saved
classID = 0
outputsFolderPath = "Datasets/Data Collect"
confidence = 0.8
save = True
blurThreshold = 35
debug = True  # Debug mode to display additional information
offsetPercentageW = 15
offsetPercentageH = 30
camWidth, camHeight = [640, 550]
floatingPoint = 6
classID = 0

# Define thresholds for face verification
min_face_area = 5000  # Minimum area for face
aspect_ratio_threshold = 0.75  # Typical aspect ratio for real faces
brightness_threshold = 150  # Max brightness for mobile screen detection (adjust based on tests)

# Ensure the output directory exists, create if not
if not os.path.exists(outputsFolderPath):
    os.makedirs(outputsFolderPath)

# Initialize the webcam, use 0 or 1 depending on the camera (default is 0)
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

# Initialize the FaceDetector object
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

listInfo = []  # Initialize the list outside the loop to collect face information

# Run the loop to continually get frames from the webcam
while True:
    # Read the current frame from the webcam
    success, img = cap.read()

    # Create a copy of the image before any drawing for saving clean images
    clean_img = img.copy()

    listBlur = []  # Initialize listBlur within the loop

    if not success:
        break

    # Detect faces in the image
    img, bboxs = detector.findFaces(img, draw=False)

    # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = float(bbox['score'][0] * 100)

            # Check the score and only process if confidence is higher than threshold
            if score > confidence * 100:
                # Calculate face area and aspect ratio
                face_area = w * h
                aspect_ratio = h / w  # Height-to-width ratio

                # Extract the face region and calculate its brightness
                imgFace = img[y:y + h, x:x + w]
                brightness = int(np.mean(imgFace)) if imgFace.size > 0 else 0

                # Determine if the detected face is "Real" or "Fake"
                if face_area > min_face_area and aspect_ratio > aspect_ratio_threshold and brightness < brightness_threshold:
                    label = "Real"
                else:
                    label = "Fake"

                # Calculate the offsets for width and height
                offsetW = (offsetPercentageW / 100) * w
                offsetH = (offsetPercentageH / 100) * h

                # Adjust the bounding box based on the new offsets
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                y = int(y - offsetH * 3)  # Tripling the height offset for y
                h = int(h + offsetH * 3.5)  # Increase height by 3.5 times the offset

                # Adding Failsafe for boundary conditions
                if x < 0: x = 0
                if y < 0: y = 0
                if x + w > img.shape[1]: w = img.shape[1] - x
                if y + h > img.shape[0]: h = img.shape[0] - y

                # Find Blurriness
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var()) if imgFace.size > 0 else 0

                # Check if blur value exceeds threshold
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # Normalize Values
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2
                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)

                # Limit the values to 1 to avoid exceeding bounds
                xcn = min(xcn, 1)
                ycn = min(ycn, 1)
                wn = min(wn, 1)
                hn = min(hn, 1)

                print(f'Normalized Center: {xcn}, {ycn}, Width: {wn}, Height: {hn}')

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # Draw Data
                color = (0, 255, 0) if label == "Real" else (0, 0, 255)  # Green for Real, Red for Fake
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                cvzone.putTextRect(img, f'{label}', (x, y - 20), scale=2, thickness=2, offset=5, colorT=color)
                cvzone.cornerRect(img, (x, y, w, h), colorC=color)

                if debug:
                    cvzone.putTextRect(img, f'Debug: Blur {blurValue} Score {score:.2f}% Area {face_area} Brightness {brightness} Aspect Ratio {aspect_ratio:.2f}',
                                       (x, y - 60), scale=1, thickness=2, offset=5)

    # Display the image in a window named 'Image'
    cv2.imshow("Image", img)

    # Press 'q' to break the loop and close the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if save is enabled and if all images are clear (non-blurry)
    if save:
        if all(listBlur) and listBlur:
            timeNow = str(time()).replace('.', '_')
            filePath = os.path.join(outputsFolderPath, f"{timeNow}.jpg")
            print(f"Saving image at: {filePath}")
            cv2.imwrite(filePath, clean_img)

# Save the collected information into a text file
with open("test.txt", 'a') as f:
    for info in listInfo:
        f.write(info)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
