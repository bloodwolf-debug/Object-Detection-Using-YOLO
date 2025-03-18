from ultralytics import YOLO
import cv2

# Load YOLOv8 Nano model (fastest for real-time, tradeoff with accuracy)
model = YOLO("yolov8n.pt")  

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        break  # Stop if there's an issue

    results = model(frame)  # Run YOLO detection on the frame

    annotated_frame = results[0].plot()  # Draw detections on the frame

    cv2.imshow("Live Detection", annotated_frame)  # Show the live video with detections

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
