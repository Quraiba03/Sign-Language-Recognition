import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Prompt user for label
label = input("Enter the label for which you want to save images/videos: ").strip()
folder = os.path.join(r"C:\Users\Muniq\OneDrive\Desktop\Sign-Language-detection-main\Data", label)
os.makedirs(folder, exist_ok=True)
print(f"Saving data to folder: {folder}")

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_path = os.path.join(folder, f"{label}_video.avi")
out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

print("Press 's' to save images, 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image from the camera.")
        break

    hands, img = detector.findHands(img)
    out.write(img)  # Write the frame to the video file

    if hands:
        for i, hand in enumerate(hands):  # Loop through each detected hand
            x, y, w, h = hand['bbox']

            # Ensure valid cropping coordinates
            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:  # Check if cropping was successful
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                aspectRatio = h / w if w != 0 else 0

                # Calculate aspect ratio and resize
                if aspectRatio > 1:  # Height > Width
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = (imgSize - wCal) // 2
                    imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
                elif aspectRatio > 0:  # Width > Height
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = (imgSize - hCal) // 2
                    imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

                # Display cropped and processed images
                cv2.imshow(f'ImageCrop Hand {i + 1}', imgCrop)
                cv2.imshow(f'ImageWhite Hand {i + 1}', imgWhite)

                # Save images on key press
                if cv2.waitKey(1) & 0xFF == ord("s"):
                    counter += 1
                    filename = os.path.join(folder, f"Hand_{i + 1}_Image_{time.time()}.jpg")
                    cv2.imwrite(filename, imgWhite)
                    print(f"Saved Image {counter}: {filename}")

    # Display original image
    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key press
        print("Exiting...")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
