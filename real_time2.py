from ultralytics import YOLO
import cv2

# Load YOLOv8 Nano model (fastest for real-time detection)
model = YOLO("yolov8m.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    results = model(frame)  # Run YOLO detection
    annotated_frame = results[0].plot()  # Draw detections on the frame

    cv2.imshow("Live YOLOv8 Detection", annotated_frame)  # Show the live window

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
