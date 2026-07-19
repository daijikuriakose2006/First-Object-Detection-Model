from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # URL or local path to the image
    img_url = "https://ultralytics.com/images/bus.jpg"

    # Run inference on the image
    # We set save=True to save the annotated image, or we can manually save it.
    # YOLO's predict method allows passing save=True and we can specify the name/path.
    results = model.predict(img_url)

    # Let's save the annotated image to results.jpg
    for result in results:
        # Save the result to results.jpg in the current working directory
        result.save(filename="results.jpg")

    print("Inference completed! Output saved to results.jpg")

if __name__ == "__main__":
    main()
