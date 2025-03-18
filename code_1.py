from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8x.pt")

# Run inference on an image
image_path = "getty_524578371_138866.jpg"
results = model(image_path)  # This returns a list

# Extract the first result
result = results[0]

# Show results using OpenCV
annotated_image = result.plot()  # Draw bounding boxes
cv2.imshow("YOLOv8 Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
