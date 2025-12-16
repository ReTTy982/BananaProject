import image_loader, image_processing
import cv2
import numpy as np

image_processing.load_presets()

def build_view(image):
    unripe_mask = image_processing.create_unripe(image)
    ripe_mask = image_processing.create_ripe(image)
    overripe_mask = image_processing.create_overripe(image)
    total_mask = image_processing.create_total_mask(image)

    unripe_mask = cv2.cvtColor(unripe_mask, cv2.COLOR_GRAY2BGR)
    ripe_mask = cv2.cvtColor(ripe_mask, cv2.COLOR_GRAY2BGR)
    overripe_mask = cv2.cvtColor(overripe_mask, cv2.COLOR_GRAY2BGR)
    total_mask = cv2.cvtColor(total_mask, cv2.COLOR_GRAY2BGR)

    empty = np.zeros_like(image)

    top = np.hstack([image, total_mask, empty])
    bottom = np.hstack([unripe_mask, ripe_mask, overripe_mask])

    return np.vstack([top, bottom])

if __name__ == "__main__":
    image = image_loader.load_next()
    cv2.namedWindow("Banana analysis", cv2.WINDOW_NORMAL)

    while True:
        combined = build_view(image)
        cv2.imshow("Banana analysis", combined)

        key = cv2.waitKeyEx(0)

        if key == ord('q'):
            break

        elif key == 2555904:  
            image = image_loader.load_next()

        elif key == 2424832:  
            image = image_loader.load_previous()
        
        elif key == ord('r'):
            presets = image_processing.load_presets()

    cv2.destroyAllWindows()