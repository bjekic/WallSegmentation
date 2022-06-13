import numpy as np
import torch
import PIL
from PIL import Image


def imresize(im, size, interp='bilinear'):
    """
        Function for image resizing with given interpolation method
    """
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


def accuracy(preds, label): #TODO check difference with pixel_acc
    """
        Function for calculating pixel accuracy of an image
    """
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def pixel_acc(pred, label):
    """
        Function for calculating the pixel accuracy between the predicted image and labeled image
    """
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0)  # some labels are -1 and are ignored
    acc_sum = (valid * (preds == label)).sum()
    pixel_sum = valid.sum()
    return acc_sum / (pixel_sum + 1e-10)


def IOU(pred, labels):
    """
        Function for calculating IOU of an image
    """
    _, preds = torch.max(pred, dim=1)
    intersection = ((preds == 0) * (labels == 0)).sum()
    union = ((preds == 0) + (labels == 0)).sum() + 1e-15  # protection from division with 0
    return intersection / union


def visualize_wall(img, pred):
    """
        Function for visualizing wall prediction 
        (original image, segmentation mask and original image with the segmented wall)
    """
    img_green = img.copy()
    black_green = img.copy()
    img_green[pred == 0] = [0, 255, 0]
    black_green[pred == 0] = [0, 255, 0]
    black_green[pred != 0] = [0, 0, 0]
    im_vis = np.concatenate((img, black_green, img_green), axis=1)
    PIL.Image.fromarray(im_vis).show()


def not_None_collate(x):
    return x
