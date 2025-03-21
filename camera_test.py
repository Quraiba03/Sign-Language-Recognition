import cv2

# Open the default camera
cap = cv2.VideoCapture(0)

# Try to read a frame from the camera
ret, frame = cap.read()

# Check if the frame was successfully captured
print(ret)  # Should print True if the camera is working

# Release the camera
cap.release()
