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