import os
import cv2
INDEX = 0
DIRECTORY_NAME = "./Assets/"
FILE_LIST = os.listdir(DIRECTORY_NAME)
WIDTH = 640
HEIGHT = 480

def load_next():
    global INDEX
    
    if INDEX >= len(FILE_LIST):
        INDEX = 0
    file_name = FILE_LIST[INDEX]
    INDEX += 1
    image = cv2.imread(f"{DIRECTORY_NAME}{file_name}")
    image = cv2.resize(image, (WIDTH, HEIGHT))
    return image

def load_previous():
    global INDEX

    if INDEX < 0:
        INDEX = len(FILE_LIST) - 1
    file_name = FILE_LIST[INDEX]
    INDEX -= 1
    image = cv2.imread(f"{DIRECTORY_NAME}{file_name}")
    image = cv2.resize(image, (WIDTH, HEIGHT))
    return image



