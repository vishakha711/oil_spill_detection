import csv
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from ml_layer.georeference import bbox_to_geojson_polygon
from ml_layer.export_payload import build_backend_payload

HISTORICAL_DATA_DIR = Path("./data/oil/water")
JSON_OUTPUT_DIR = Path("./data/json_outputs/water")
MODEL_PATH = Path("./ml_layer/weights/best.onnx")

CLOUD_NAME = "iza4thnt"
CLOUDINARY_FOLDER = "oil_spill_detection"

JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model = YOLO(str(MODEL_PATH))

historical_metadata = {}
try:
    with open("./data/data_table.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["IMAGE (jpg_file)"]
            historical_metadata[img_name] = row["Date/Time (start_time)"]
except FileNotFoundError:
    print("Warning: data_table.csv not found. Using fallback timestamps.")
except KeyError as e:
    print(f"Error: Column {e} not found in CSV. Check your exact column headers.")


def process_historical_batch():
    for img_path in HISTORICAL_DATA_DIR.glob("*.jpg"):

        inference_results = model(str(img_path), conf=0.50)
        corners = get_historical_corners(img_path.name)
        historical_time = extract_timestamp(img_path.name)

        image_detections = []

        for r in inference_results:
            img_height, img_width = r.orig_shape

            live_image_url = f"https://res.cloudinary.com/{CLOUD_NAME}/image/upload/{CLOUDINARY_FOLDER}/{img_path.name}"

            for box in r.boxes:
                bbox_pixels = box.xyxy[0].tolist()
                confidence = float(box.conf[0])

                geojson_poly = bbox_to_geojson_polygon(
                    bbox_pixels,
                    img_width,
                    img_height,
                    corners
                )

                wgs84_coords = geojson_poly["coordinates"][0]

                payload = build_backend_payload(
                    wgs84_coords,
                    historical_time,
                    confidence,
                    live_image_url
                )

                image_detections.append(payload)

        if image_detections:
            output_filename = JSON_OUTPUT_DIR / f"{img_path.stem}.json"

            with open(output_filename, "w") as json_file:
                json.dump({"detections": image_detections}, json_file, indent=4)


def get_historical_corners(filename):
    return {
        "ul": (24.0000, 35.1000),
        "ur": (24.1000, 35.1000),
        "bl": (24.0000, 35.0000),
        "br": (24.1000, 35.0000)
    }


def extract_timestamp(filename):
    raw_time_str = historical_metadata.get(
        filename,
        "1970-01-01T00:00:00"
    )

    try:
        return datetime.strptime(
            raw_time_str,
            "%Y-%m-%dT%H:%M:%S"
        )

    except ValueError:
        print(
            f"Warning: Could not parse date format '{raw_time_str}' "
            f"for {filename}. Using default time."
        )
        return datetime.utcnow()


if __name__ == "__main__":
    process_historical_batch()