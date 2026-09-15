import json
import uuid
import math
from datetime import datetime
from pyproj import Geod
from shapely.geometry import Polygon


def estimate_spill_age_heuristic(area_m2, perimeter_m):
    if perimeter_m == 0:
        return 2.0

    compactness = (4 * math.pi * area_m2) / (perimeter_m ** 2)
    compactness = max(0.01, min(compactness, 1.0))
    age_hours = 72 * (1 - compactness)

    return max(2.0, min(round(age_hours, 1), 72.0))


def build_backend_payload(wgs84_polygon_coords, acquisition_time, confidence, image_reference):
    poly = Polygon(wgs84_polygon_coords)
    centroid_lon, centroid_lat = poly.centroid.x, poly.centroid.y

    geod = Geod(ellps="WGS84")
    area_m2, perimeter_m = geod.geometry_area_perimeter(poly)
    area_km2 = abs(area_m2) / 1_000_000

    estimated_age = estimate_spill_age_heuristic(area_m2, perimeter_m)

    return {
        "spill_id": f"spill_{uuid.uuid4().hex[:6]}",
        "detected_at": acquisition_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "centroid": {
            "lon": round(centroid_lon, 4),
            "lat": round(centroid_lat, 4)
        },
        "polygon": [[round(lon, 4), round(lat, 4)] for lon, lat in wgs84_polygon_coords],
        "area_km2": round(area_km2, 2),
        "estimated_age_hours": estimated_age,
        "confidence_score": round(float(confidence), 2),
        "image_reference": image_reference
    }


if __name__ == "__main__":
    mock_coords = [
        [-88.3100, 28.5100],
        [-88.3080, 28.5150],
        [-88.3000, 28.5140],
        [-88.3100, 28.5100]
    ]

    mock_time = datetime.utcnow()
    mock_image_url = "https://res.cloudinary.com/iza4thnt/image/upload/v1789433034/oil_spill_detection/bugfah8uwwwskh8ipwsu.jpg"

    final_json = json.dumps(
        build_backend_payload(mock_coords, mock_time, 0.8754, mock_image_url),
        indent=2
    )

    print(final_json)