import image_loader, image_processing
import cv2




if __name__ == "__main__":
    image = image_loader.load_next()
    unripe_mask = image_processing.create_unripe(image)
    ripe_mask = image_processing.create_ripe(image)
    overripe_mask = image_processing.create_overripe(image)
    total_mask = image_processing.create_total_mask(image)


    cv2.imshow("Original",image)
    cv2.imshow("Unripe Mask",unripe_mask)
    cv2.imshow("Ripe Mask",ripe_mask)
    cv2.imshow("Overripe Mask",overripe_mask)
    cv2.imshow("Total Mask",total_mask)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()
        