import image_loader, clahe,segmentation,cluster_selector
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img = image_loader.load_specific(0)
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

