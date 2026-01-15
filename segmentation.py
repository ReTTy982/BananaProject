import numpy as np
import matplotlib.pyplot as plt
import cv2
import clahe

def segmentation(image,k):
    pixel_vals = image.reshape((-1,3))

    pixel_vals = np.float32(pixel_vals)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    segmented_image = segmented_data.reshape((image.shape))
    return segmented_image

if __name__ == "__main__":
    img = cv2.imread('Assets/banana1.jpg')
    image_clahe = clahe.process_clahe(img,1)
    segmented_image = segmentation(image_clahe,2)
    plt.imshow(segmented_image)
    plt.show()