import image_loader, clahe,segmentation,cluster_selector, stats
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #stats.process_folders_in_batches(base_path="./Assets/bananadataset/dataset/data", batch_size=200)
    #stats.show_random_peak_masks(base_path="./Assets/bananadataset/dataset/data")
    #stats.process_full_hsv_by_folder(base_path="./Assets/bananadataset/dataset/data", batch_size=200)
    #stats.verify_all_peaks(base_path="./Assets/bananadataset/dataset/data", num_samples=100)
    #stats.show_forced_peak_samples(base_path="./Assets/bananadataset/dataset/data", target_hue=75, num_to_find=10)
    #img = image_loader.load_specific(0)
    #img_clahe = clahe.process_clahe(img,1)
    #img_segmented = segmentation.segmentation(img_clahe,10)
    #semtended_image = cluster_selector.cluster_main(img_clahe,6)
    img = image_loader.load_specific(3)
    img_clahe = clahe.process_clahe(img,1)
    #img_segmented = segmentation.segmentation(img_clahe,10)
    semtended_image = cluster_selector.cluster_main(img_clahe,6)
    #cluster_selector.cluster_main(semtended_image,2)






    '''
    resize
    claha
    segmentacja
    klaster selector
    procesing danych

    '''

