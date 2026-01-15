import cv2
import numpy as np


def process_clahe(image,gamma):
    if gamma == None:
        gamma = 1.0
    print(f"GAMMA: {gamma}")
    h, w = image.shape[:2]

    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(image, mask, (0, 0), (255, 255, 255))

    cv2.floodFill(image, mask, (0, h-1), (255, 255, 255))
    cv2.floodFill(image, mask, (w-1, 0), (255, 255, 255))
    cv2.floodFill(image, mask, (w-1, h-1), (255, 255, 255))


    cv2.imshow("przed", image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("po", hsv)
    
    h, s, v = cv2.split(hsv)

    v_gamma = np.power(v / 255.0, 1.0 / gamma) * 255.0
    v_gamma = v_gamma.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(16, 16))
    v_final = clahe.apply(v_gamma)
    denoised_v = cv2.bilateralFilter(v_final, d=9, sigmaColor=75, sigmaSpace=75)
    hsv_final = cv2.merge((h, s, denoised_v))

    result_rgb = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR) # RGB zmienione na BGR
    return result_rgb

