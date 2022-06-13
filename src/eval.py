import torch
import torchvision.transforms
from PIL import Image
import numpy as np

from Utils.constants import IMAGENET_MEAN, IMAGENET_STD
from Utils.utils import IOU, visualize_wall, accuracy
from Utils.constants import DEVICE
from tqdm import tqdm


def validation_step(segmentation_module, loader, writer, epoch):
    """
        Function for evaluating the segmentation module on validation dataset
    """
    segmentation_module.eval()
    segmentation_module.to(DEVICE)
    
    total_acc = 0
    total_IOU = 0
    counter = 0
    
    for batch_data in tqdm(loader):
        batch_data = batch_data[0]

        seg_label = np.array(batch_data['seg_label'])
        segSize = (seg_label.shape[0], seg_label.shape[1])

        with torch.no_grad():
            scores = segmentation_module(batch_data, segSize=segSize)

        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()

        # calculate accuracy and IOU
        acc, _ = accuracy(pred, seg_label)
        IOU_curr = IOU(scores.cpu(), seg_label)
        total_IOU += IOU_curr
        total_acc += acc
        counter += 1

    average_acc = total_acc/counter
    average_IOU = total_IOU/counter

    writer.add_scalar('Validation set: accuracy', average_acc, epoch)
    writer.add_scalar('Validation set: IOU', average_IOU, epoch)
    
    return average_acc, average_IOU


def segment_image(segmentation_module, img, disp_image=True):
    """
        Function for segmenting wall in the input image. The input can be path to image, or a loaded image
    """
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    if isinstance(img, str):
        img = Image.open(img)
    
    img_original = np.array(img)
    img_data = pil_to_tensor(img)
    singleton_batch = {'img_data': img_data[None].to(DEVICE)}
    segSize = img_original.shape[:2]

    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=segSize)

    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()

    if disp_image:
        visualize_wall(img_original, pred)

    return pred
