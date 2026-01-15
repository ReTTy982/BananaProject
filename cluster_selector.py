from enum import Enum
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def display_image(img, title="Image"):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_RGB_image(img, title="def"):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


class BananaStage(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    BROWN = "brown"


def _cluster_preprocessing(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixel_values = img.reshape((-1, 3))  
    pixel_values = np.float32(pixel_values)  

    return  pixel_values


def _apply_kmeans(image, n_clusters):
    k = n_clusters  
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels, centers


def _show_kmeans(image, centers, labels):
    centers = np.uint8(centers)  
    segmented_image = centers[labels.flatten()]  
    segmented_image = segmented_image.reshape(image.shape)  
    display_image(segmented_image)
    return segmented_image


def _calculate_area_per_cluster(image, k, centers, labels):
    center_hues = []
    for i in range(k):
        cluster_to_isolate = i

        masked_image = np.copy(image)
        masked_image = masked_image.reshape((-1, 3))
        masked_image[labels.flatten() != cluster_to_isolate] = [0, 0, 0]
        masked_copy = masked_image.copy()
        masked_copy = masked_copy.reshape(image.shape)
        center_color = centers[i].astype(np.uint8)
        colored_cluster = np.zeros_like(image)  # (H,W,3)
        mask2d = (labels.reshape(image.shape[0], image.shape[1]) == i)

        colored_cluster[mask2d] = center_color
        stage = classify_cluster(centers[i][0],centers[i][1])
        print(f"Hue: {centers[i][0]} classified as {stage}")

        display_image(colored_cluster, f"Cluster {i} colored by center {center_color}")
        masked_image[labels.flatten() == cluster_to_isolate] = [255, 255, 255]
        masked_image = masked_image.reshape(image.shape)
        print(f"centrum dla {i}: {centers[i]}")

        white_pixels = np.count_nonzero(masked_image)


        mask = (labels.reshape(image.shape[0], image.shape[1]) == cluster_to_isolate).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Liczba białych pikseli: {white_pixels}, liczba konturów w przedziale: {len(contours)}")

        cv2.drawContours(masked_image, contours, -1, (0, 255, 0), 10)

        print(len(contours))
        center_hues.append(centers[i][0]) 
    return center_hues


def classify_cluster(h,s):
    if 35 <= h <= 65:
        return "GREEN"
    elif 25 <= h < 35:
        return "YELLOW"
    elif 10 <= h < 25:
        if s <= 70 and h >=20:
            return "YELLOW"
        return "BROWN"
    else:
        return "UNKNOWN"


def cluster_main(image, n_clusters):
    k = n_clusters
    pre = _cluster_preprocessing(image)
    labels, centers = _apply_kmeans(pre, k)
    h = _calculate_area_per_cluster(image, k, centers, labels)
    return _show_kmeans(image, centers, labels)

