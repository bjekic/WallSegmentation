import os, torch, PIL, torchvision.transforms
import numpy as np
from Models.models import SegmentationModule, build_encoder, build_decoder
from Models.dataset import ValDataset

from Utils.constants import IMAGENET_MEAN, IMAGENET_STD
from Utils.utils import pixel_acc, IOU, visualize_wall, accuracy
from Utils.constants import DEVICE, ODGT_EVALUTATION
# from tensorboard import SummaryWriter
from tqdm import tqdm

def evaluate(segmentation_module, loader, writer, epoch):
    """
        Function for evaluating the segmentation module on validation dataset
    """
    segmentation_module.eval()
    segmentation_module.to(DEVICE)
    
    total_acc = 0
    total_IOU = 0
    counter = 0
    
    for batch_data in tqdm(loader):
        batch_data = batch_data[0]  # unpack the batch to have only one image
        # seg_label = torch.unsqueeze(batch_data['seg_label'], dim=0)
        # # seg_label = np.array(batch_data['seg_label'])
        # segSize = (seg_label.shape[1], seg_label.shape[2])

        # seg_label = [seg_label]

        # with torch.no_grad():
        #     scores = segmentation_module(batch_data, segSize=segSize)
        #
        # _, pred = torch.max(scores, dim=1)
        # pred = pred.cpu()[0].numpy()  # accessing the only image using [0]

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
    

def eval_one_img(segmentation_module, device, image_folder_path, annotations_folder_path, img_number):
    """
        Function for evaluating the Segmentation Module on one image, with given root path and image number 
    """
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=IMAGENET_MEAN, # These are RGB mean and std values
            std=IMAGENET_STD)  # across a large photo dataset.
        ])
    
    img_str = str(img_number)
    img_str_padded = img_str.zfill(8)
    # img = PIL.Image.open(root_path + 'ADEChallengeData2016/images/validation/ADE_val_' + img_str_padded + '.jpg').convert('RGB')
    # segm = PIL.Image.open(root_path + 'ADEChallengeData2016/annotations/validation/ADE_val_' + img_str_padded + '.png')
    
    img = PIL.Image.open(os.path.join(image_folder_path, f"ADE_val_{img_str_padded}.jpg")).convert('RGB')
    segm = PIL.Image.open(os.path.join(annotations_folder_path, f"ADE_val_{img_str_padded}.png"))
        
    img_original = np.array(img)
    img_data = pil_to_tensor(img)
    singleton_batch = {'img_data': img_data[None].to(device)}
   
    segm = np.array(segm)-1 
    segm[segm > 0] = 1
    segSize = segm.shape
    
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=segSize)
        
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    
    acc, pix = accuracy(pred, segm)
    img_IOU = IOU(pred, segm)
    print('Accuracy is: ' + str(acc))
    print('IOU is: ' + str(img_IOU))
    visualize_wall(img_original, pred)
    
    return acc, img_IOU


def main_evaluate(segmentation_module, data_root_path, writer, epoch):
    """
        Function for evaluating Segmentation module
    """
    # TODO: Change this function to be more intuitive

    # Dataset and Loader
    dataset_val = ValDataset(data_root_path, ODGT_EVALUTATION)
    loader_val = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=1,  # it has to be 1, because images are not the same size
                                             shuffle=False,
                                             collate_fn=lambda x: x,
                                             drop_last=True)
    
    print(f'Starting evaluation after epoch {epoch}')
    avg_acc, avg_IOU = evaluate(segmentation_module, loader_val, writer, epoch)
    print('Evaluation Done!\n')
    
    return avg_acc, avg_IOU
    

# def segment_image(segmentation_module, img, device):
#     """
#         Function for segmenting wall in the input image
#     """
#     pil_to_tensor = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(
#             mean=IMAGENET_MEAN, # These are RGB mean+std values
#             std=IMAGENET_STD)  # across a large photo dataset.
#         ])
#
#     img_original = np.array(img)
#     img_data = pil_to_tensor(img)
#     singleton_batch = {'img_data': img_data[None].to(device)}
#     segSize = img_original.shape[:2]
#
#     with torch.no_grad():
#         scores = segmentation_module(singleton_batch, segSize=segSize)
#
#     _, pred = torch.max(scores, dim=1)
#     pred = pred.cpu()[0].numpy()
#
#     visualize_wall(img_original, pred)