import os
import torch
import cv2
import numpy as np

def show_torch_image(torch_image):
    for image in torch_image:
        image = image.permute(1, 2, 0)
        image = image.detach().numpy()
        cv2.imshow('111', image)
        cv2.waitKey()
    return None

def dataload(path_to_images, img_size=640):
    extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
    tree = os.walk(path_to_images)
    list_of_images = []
    for folder1, folder2, names in tree:
        for file in names:
            name, extension = [x for x in file.split('.')]

            if extension in extensions:
                if os.path.exists(folder1 + name + '.txt'):
                    print(name + '.' + extension, ':')
                    # get image from folder
                    image = cv2.imread(folder1 + name + '.' + extension)
                    if not image is None:
                        image = cv2.resize(image, (img_size, img_size))
                        print(image.shape)
                    else:
                        None
    return None