from ultralytics import YOLO

def main():
    print("Downloading YOLOv8n model...")
    # This automatically downloads yolov8n.pt if not found in the local directory
    model = YOLO("yolov8n.pt")
    print("Model downloaded successfully!")

if __name__ == "__main__":
    main()
