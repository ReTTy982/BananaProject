import os
import cv2
INDEX = 0
DIRECTORY_NAME = "./Assets/"
FILE_LIST = sorted(os.listdir(DIRECTORY_NAME))
WIDTH = 640
HEIGHT = 480

def _load_current():
    file_name = FILE_LIST[INDEX]
    image = cv2.imread(os.path.join(DIRECTORY_NAME, file_name))
    image = cv2.resize(image, (WIDTH, HEIGHT))
    return image


def load_next():
    global INDEX
    INDEX = (INDEX + 1) % len(FILE_LIST)
    return _load_current()


def load_previous():
    global INDEX
    INDEX = (INDEX - 1) % len(FILE_LIST)
    return _load_current()

def load_specific(index):
    global INDEX
    if 0 <= index < len(FILE_LIST):
        INDEX = index
    return _load_current()


