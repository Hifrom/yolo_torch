import torch
import cv2
import os
import argparse
from utils.model import yolov4

# === device ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === image ===
img = cv2.resize(cv2.imread('./test.jpg'), (224, 224))
# cv2.imshow('1', img)
# cv2.waitKey()
image = torch.from_numpy(img) / 255.0
image = image.permute(2, 0, 1).unsqueeze(0).to(device)
print('image.shape before: {0}'.format(image.shape))

# === model ===
model = yolov4()
model.to(device)
out = model.forward(image)
print('image.shape after: {0}'.format(out.shape))
