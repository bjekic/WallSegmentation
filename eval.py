import os, torch, PIL, torchvision.transforms
import numpy as np
from Models.models import SegmentationModule, build_encoder, build_decoder
from Models.dataset import ValDataset
from IPython import display


def accuracy(preds, label):
    """
        Function for calculating pixel accuracy of an image
    """
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def IOU(preds: np.array, labels: np.array):
    """
        Function for calculating IOU of an image
    """
    intersection = sum(sum((preds==0) * (labels==0)))
    union = sum(sum((preds==0) + (labels==0))) + 1e-15 # protection from division with 0
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
    display(PIL.Image.fromarray(im_vis))


def evaluate(segmentation_module, loader):
    """
        Function for evaluating the segmentation module on validation dataset
    """
    segmentation_module.eval()
    segmentation_module.cuda()
    
    total_acc = 0
    total_IOU = 0
    counter = 0
    max_IOU = 0
    
    for batch_data in loader:
        batch_data = batch_data[0]
        seg_label = np.array(batch_data['seg_label'])
        img = batch_data['img_data']
        segSize = (seg_label.shape[0], seg_label.shape[1])
        
        with torch.no_grad():
            scores = segmentation_module(batch_data, segSize=segSize)
                
        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()

        # calculate accuracy and IOU
        acc, pix = accuracy(pred, seg_label)
        IOU_curr = IOU(pred, seg_label)
        total_IOU += IOU_curr
        total_acc += acc
        counter += 1
        
        # Finding the image with maximum IOU in validation dataset
        if IOU_curr > max_IOU:
            max_IOU_img = batch_data['name']
            max_IOU = IOU_curr
        
        # Printing average accuracy and average IOU every 100 iterations
        if counter % 100 == 0 and counter > 0:
            print(str(counter) + ' iteration: Average accuracy: ' + str(total_acc/counter) + '; Average IOU: ' + str(total_IOU/counter) + ';')
            
    average_acc = total_acc/counter
    average_IOU = total_IOU/counter
    
    print('[Eval Summary]:')
    print('Average accuracy on validation set is: ' + str(average_acc) + '; Average IOU on validation set is: ' + str(average_IOU) + '.')
    
    return average_acc, average_IOU, max_IOU_img, max_IOU
    

def eval_one_img(segmentation_module, root_path, img_number):
    """
        Function for evaluating the Segmentation Module on one image, with given root path and image number 
    """
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean and std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
        ])
    
    img_str = str(img_number)
    img_str_padded = img_str.zfill(8)
    img = PIL.Image.open(root_path + 'ADEChallengeData2016/images/validation/ADE_val_' + img_str_padded + '.jpg').convert('RGB')
    segm = PIL.Image.open(root_path + 'ADEChallengeData2016/annotations/validation/ADE_val_' + img_str_padded + '.png')
        
    img_original = np.array(img)
    img_data = pil_to_tensor(img)
    singleton_batch = {'img_data': img_data[None].cuda()}
   
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


def main_evaluate(transfer_learn_all=False, transfer_learn_decoder=False):
    """
        Function for evaluating Segmentation module. Choice of the Segmentation Module is done using input Bool variables.
    """
    # TODO: Change this function to be more intuitive
    net_encoder = build_encoder(pretrained=False, train_only_wall=True)
    net_decoder = build_decoder(pretrained=False, train_only_wall=True)
    
    if transfer_learn_all:        
        weights_encoder = 'Model weights/wall_encoder_epoch_20.pth'
        weights_decoder = 'Model weights/wall_decoder_epoch_20.pth'
        
    elif transfer_learn_decoder:
        weights_encoder = 'Model weights/transfer_encoder.pth'
        weights_decoder = 'Model weights/transfer_decoder.pth'
        
    else:
        weights_encoder = 'Model weights/Output_only_encoder.pth'
        weights_decoder = 'Model weights/Output_only_decoder.pth'
    
    net_encoder.load_state_dict(torch.load(weights_encoder, map_location=lambda storage, loc: storage), strict=False)
    net_decoder.load_state_dict(torch.load(weights_decoder, map_location=lambda storage, loc: storage), strict=False)
    
    # Creating Segmentation Module
    segmentation_module = SegmentationModule(net_encoder, net_decoder)
    
    # Dataset and Loader
    dataset_val = ValDataset("data/", "data/validation.odgt")
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        drop_last=True)
    
    print('Starting evaluation')
    avg_acc, avg_IOU, max_IOU_img, max_IOU =  evaluate(segmentation_module, loader_val)
    print('Evaluation Done!')
    
    return avg_acc, avg_IOU, max_IOU_img, max_IOU
    

def segment_image(segmentation_module, img):
    """
        Function for segmenting wall in the input image
    """
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
        ])
    
    img_original = np.array(img)
    img_data = pil_to_tensor(img)
    singleton_batch = {'img_data': img_data[None].cuda()}
    segSize = img_original.shape[:2]
    
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=segSize)
        
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    
    visualize_wall(img_original, pred)