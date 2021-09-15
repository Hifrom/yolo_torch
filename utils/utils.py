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
    list_of_labels = []
    list_of_broken_images = []
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
                        if list_of_images == []:
                            list_of_images = np.expand_dims(image, axis=0)
                            # txt labels
                            with open(folder1 + name + '.txt') as txtfile:
                                flag = 0
                                for line in txtfile:
                                    cls, x, y, w, h = [x for x in line.split()]
                                    if flag == 0:
                                        list_of_labels_local = np.array([[float(cls), float(x), float(y), float(w), float(h)]])
                                        # print(f'list_of_labels_local: {list_of_labels_local}')
                                    else:
                                        list_of_labels_local = np.concatenate((list_of_labels_local, np.array([[float(cls), float(x), float(y), float(w), float(h)]])), axis = 0)
                                        # print(f'list_of_labels_local: {list_of_labels_local}')
                                    flag = 1
                                if list_of_labels == []:
                                    list_of_labels = np.expand_dims(list_of_labels_local, axis=0)
                                else:
                                    list_of_labels = np.concatenate((list_of_labels, list_of_labels_local), axis=0)
                            print('list_of_labels_1_image', list_of_labels.shape)
                        else:
                            list_of_images = np.concatenate((list_of_images, np.expand_dims(image, axis=0)), axis=0)
                            with open(folder1 + name + '.txt') as txtfile:
                                flag = 0
                                for line in txtfile:
                                    cls, x, y, w, h = [x for x in line.split()]
                                    if flag == 0:
                                        list_of_labels_local = np.array([[float(cls), float(x), float(y), float(w), float(h)]])
                                    else:
                                        list_of_labels_local = np.concatenate((list_of_labels_local, np.array([[float(cls), float(x), float(y), float(w), float(h)]])), axis = 0)
                                    flag = 1
                                if list_of_labels == []:
                                    print('list_of_labels_another_image', list_of_labels.shape)
                                    list_of_labels = np.expand_dims(list_of_labels_local, axis=0)
                                else:
                                    print('list_of_labels_another_image', list_of_labels.shape)
                                    print('list_of_labels_local', list_of_labels_local.shape)
                                    list_of_labels = np.concatenate((list_of_labels, np.expand_dims(list_of_labels_local, axis=0)), axis=0)
                    else:
                        list_of_broken_images.append(folder1 + name + '.' + extension)
                else:
                    # add images without annotation
                    None
    return list_of_images, list_of_labels