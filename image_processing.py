import cv2
import json
from enum import Enum

# # Niedojrzały (zielony)
# GREEN_LOWER = (30, 80, 80)
# GREEN_UPPER = (65, 255, 255)

# # Dojrzały (żółty)
# YELLOW_LOWER = (20, 100, 150)
# YELLOW_UPPER = (35, 255, 255)

# # Przejrzały (brązowy)
# BROWN_LOWER = (10, 30, 50)
# BROWN_UPPER = (25, 150, 180)

PRESET_FILE = "hsv.json"
PRESETS = None


class BananaStage(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    BROWN = "brown"


def _get(stage: BananaStage):
    return PRESETS[stage.value]


def load_presets():
    global PRESETS
    with open(PRESET_FILE, "r") as f:
        PRESETS = json.load(f)


def create_hsv_mask(image, lower, upper):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(imgHSV, tuple(lower), tuple(upper))

def create_unripe(image):
    p = _get(BananaStage.GREEN)
    return create_hsv_mask(image, p["lower"], p["upper"])


def create_ripe(image):
    p = _get(BananaStage.YELLOW)
    return create_hsv_mask(image, p["lower"], p["upper"])


def create_overripe(image):
    p = _get(BananaStage.BROWN)
    return create_hsv_mask(image, p["lower"], p["upper"])


def create_total_mask(image):
    ranges = [_get(stage) for stage in BananaStage]

    lower = [min(r["lower"][i] for r in ranges) for i in range(3)]
    upper = [max(r["upper"][i] for r in ranges) for i in range(3)]

    return create_hsv_mask(image, lower, upper)
 

