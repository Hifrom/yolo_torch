import torch
import cv2
import os
import argparse
from utils.model import yolov4
from utils.utils import show_torch_image
from utils.utils import dataload
# === device ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === image ===
img = cv2.resize(cv2.imread('./test.jpg'), (628, 628))
# cv2.imshow('1', img)
# cv2.waitKey()
image = torch.from_numpy(img) / 255.0
image = image.permute(2, 0, 1).unsqueeze(0).to(device)
# print('image.shape before: {0}'.format(image.shape))

# === dataset loader ===
path = os.path.join('./data/')
imgsz = 224
dataset = dataload(path_to_images=path, img_size=224)

# === model ===
model = yolov4()
model.to(device)
out = model.forward(image)
''' # View Image From Torch to Opencv
show_torch_image(out.to('cpu'))
'''