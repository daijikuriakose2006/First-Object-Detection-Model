from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # URL or local path to the image
    img_url = "https://ultralytics.com/images/bus.jpg"


    results = model.predict(img_url)

    # Let's save the annotated image to results.jpg
    for result in results:
        # Save the result to results.jpg in the current working directory
        result.save(filename="results.jpg")

if __name__ == "__main__":
    main()
