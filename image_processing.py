import cv2


# Niedojrzały (zielony)
GREEN_LOWER = (30, 80, 80)
GREEN_UPPER = (65, 255, 255)

# Dojrzały (żółty)
YELLOW_LOWER = (20, 100, 150)
YELLOW_UPPER = (35, 255, 255)

# Przejrzały (brązowy)
BROWN_LOWER = (10, 30, 50)
BROWN_UPPER = (25, 150, 180)




def create_hsv_mask(image,lower, upper):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, lower, upper)
    return mask

def calcullate_total_range():
    lower = [GREEN_LOWER, YELLOW_LOWER, BROWN_LOWER]
    upper = [GREEN_UPPER, YELLOW_UPPER, BROWN_UPPER]
    global_lower = tuple(min(values) for values in zip(*lower))
    global_upper = tuple(max(values) for values in zip(*upper))
    return [global_lower, global_upper]

def create_unripe(image): return create_hsv_mask(image, GREEN_LOWER, GREEN_UPPER)
def create_ripe(image): return create_hsv_mask(image, YELLOW_LOWER, YELLOW_UPPER)
def create_overripe(image): return create_hsv_mask(image, BROWN_LOWER, BROWN_UPPER)
def create_total_mask(image):
    total_range = calcullate_total_range()
    return create_hsv_mask(image, total_range[0], total_range[1])
 

