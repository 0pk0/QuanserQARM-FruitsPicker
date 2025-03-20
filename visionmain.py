from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('/Users/praveenk/Downloads/yolo_fruits_and_vegetables_v3.pt')

# Open the camera (replace 0 with your camera index if needed)
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Display the results
    results.show()

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
