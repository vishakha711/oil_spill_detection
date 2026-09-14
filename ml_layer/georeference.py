class CornerValidationError(ValueError):
    """Raised when patch corner coordinates are missing or invalid."""


def bbox_to_geojson_polygon(bbox_pixels, patch_width, patch_height, corners):
    required = {"ul", "ur", "bl", "br"}

    if set(corners) != required:
        raise CornerValidationError(f"corners must have exactly keys {sorted(required)}")

    if patch_width <= 0 or patch_height <= 0:
        raise CornerValidationError("patch_width and patch_height must be positive")

    xmin, ymin, xmax, ymax = map(float, bbox_pixels)

    if xmin > xmax or ymin > ymax:
        raise ValueError("Invalid bounding box coordinates")

    ul_lon, ul_lat = corners["ul"]
    ur_lon, ur_lat = corners["ur"]
    bl_lon, bl_lat = corners["bl"]
    br_lon, br_lat = corners["br"]

    def pixel_to_coords(x, y):
        nx = x / patch_width
        ny = y / patch_height

        top_lon = ul_lon + nx * (ur_lon - ul_lon)
        bottom_lon = bl_lon + nx * (br_lon - bl_lon)

        top_lat = ul_lat + nx * (ur_lat - ul_lat)
        bottom_lat = bl_lat + nx * (br_lat - bl_lat)

        lon = top_lon + ny * (bottom_lon - top_lon)
        lat = top_lat + ny * (bottom_lat - top_lat)

        return [round(lon, 6), round(lat, 6)]

    top_left = pixel_to_coords(xmin, ymin)
    top_right = pixel_to_coords(xmax, ymin)
    bottom_right = pixel_to_coords(xmax, ymax)
    bottom_left = pixel_to_coords(xmin, ymax)

    return {
        "type": "Polygon",
        "coordinates": [[
            top_left,
            top_right,
            bottom_right,
            bottom_left,
            top_left
        ]]
    }