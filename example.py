

import cv2
import numpy as np

HEIGHT = 480
WIDTH = 640
FILE_NAME = "bananaO1.jpg"

# Niedojrzały (zielony)
green_lower = (30, 80, 80)
green_upper = (65, 255, 255)

# Dojrzały (żółty)
yellow_lower = (20, 100, 150)
yellow_upper = (35, 255, 255)

# Przejrzały (brązowy)
brown_lower = (10, 30, 50)
brown_upper = (25, 150, 180)

# Empty callback function required for trackbar creation
def empty(a):
    pass

# Create a window to hold all HSV trackbars
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 240)

# Create trackbars for Hue, Saturation, and Value range limits
cv2.createTrackbar("HUE MIN", "Trackbars", 30, 179, empty)
cv2.createTrackbar("HUE MAX", "Trackbars", 65, 179, empty)
cv2.createTrackbar("SAT MIN", "Trackbars", 30, 255, empty)
cv2.createTrackbar("SAT MAX", "Trackbars", 255, 255, empty)
cv2.createTrackbar("VAL MIN", "Trackbars", 50, 255, empty)
cv2.createTrackbar("VAL MAX", "Trackbars", 255, 255, empty)

while True:
    # Load the image and convert it to HSV color space
    image = cv2.imread(f"./Assets/{FILE_NAME}")
    image = cv2.resize(image, (WIDTH, HEIGHT))

    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Read current positions of all trackbars
    h_min = cv2.getTrackbarPos("HUE MIN", "Trackbars")
    h_max = cv2.getTrackbarPos("HUE MAX", "Trackbars")
    s_min = cv2.getTrackbarPos("SAT MIN", "Trackbars")
    s_max = cv2.getTrackbarPos("SAT MAX", "Trackbars")
    v_min = cv2.getTrackbarPos("VAL MIN", "Trackbars")
    v_max = cv2.getTrackbarPos("VAL MAX", "Trackbars")

    # Define lower and upper HSV bounds
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Create a binary mask where white = in range, black = out of range
    mask = cv2.inRange(imgHSV, lower, upper)


    # Display original, HSV, and masked images
    cv2.imshow("Original Image", image)
    cv2.imshow("HSV Image", imgHSV)
    cv2.imshow("Mask Image", mask)

    # We got the orange part of the car image at HSV range:
    # HUE: 0 to 18, SAT: 13 to 255, VAL: 125 to 255
    # These values will make the orange part appear white in the mask, and all other areas black.
    # (Now move to color_detection_second.py to extract the actual orange region using bitwise AND)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

