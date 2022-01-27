import os
import torch
import torch.nn as nn
from Utils.constants import TOTAL_NUM_ITER, NUM_ITER_PER_EPOCH, NUM_EPOCHS, OPTIMIZER_PARAMETERS, DEVICE, NUM_WORKERS, ODGT_TRAINING, BATCH_PER_GPU
from Utils.utils import pixel_acc
from tqdm import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from Models.models import SegmentationModule, build_encoder, build_decoder
from Models.dataset import TrainDataset


def train_one_epoch(segmentation_module, iterator, optimizers, epoch, crit, writer):
    """
        Function for training model for only one epoch
    """

    segmentation_module.train()

    for i in tqdm(range(NUM_ITER_PER_EPOCH)):
        # load a batch of data
        batch_data = next(iterator)[0] #TODO check iterator (it is because the batch size in the dataloader is 1, but the batch is created in TrainDataset
        segmentation_module.zero_grad() 
        
        # adjust learning rate (learning rate "poly") #TODO change to learning rate scheduler
        cur_iter = i + (epoch - 1) * NUM_ITER_PER_EPOCH
        adjust_learning_rate(optimizers, cur_iter)
                
        # forward pass
        pred = segmentation_module(batch_data)

        # Calculate loss and accuracy
        loss = crit(pred, batch_data['seg_label'].to(DEVICE))
        acc = pixel_acc(pred, batch_data['seg_label'].to(DEVICE))
               
        loss = loss.mean()
        acc = acc.mean()

        # Backward pass
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # update average loss and acc
        writer.add_scalar('Trainig loss', loss.data.item(), (epoch - 1) * NUM_ITER_PER_EPOCH + i)
        writer.add_scalar('Trainig accuracy', acc.data.item(), (epoch - 1) * NUM_ITER_PER_EPOCH + i)


def checkpoint(nets, epoch, checkpoint_dir_path, is_best_epoch): 
    """
        Function for saving encoder and decoder weights into a file
    """
    print('Saving checkpoints...')
    net_encoder, net_decoder, _ = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()
    
    torch.save(dict_encoder, os.path.join(checkpoint_dir_path, f'encoder_epoch_{epoch}.pth'))
    torch.save(dict_decoder, os.path.join(checkpoint_dir_path, f'decoder_epoch_{epoch}.pth'))
    
    previous_encoder_epoch = os.path.join(checkpoint_dir_path, f'encoder_epoch_{epoch-1}.pth')
    if os.path.exists(previous_encoder_epoch):
        os.remove(previous_encoder_epoch)
        
    previous_decoder_epoch = os.path.join(checkpoint_dir_path, f'decoder_epoch_{epoch-1}.pth')
    if os.path.exists(previous_decoder_epoch):
        os.remove(previous_decoder_epoch)
    
    if is_best_epoch:
        prev_best_models = [os.path.join(checkpoint_dir_path, x) for x in os.listdir(checkpoint_dir_path) if x.startswith('best_')]
        for model_path in prev_best_models:
            os.remove(model_path)
        torch.save(dict_encoder, os.path.join(checkpoint_dir_path, f'best_encoder_epoch_{epoch}.pth'))
        torch.save(dict_decoder, os.path.join(checkpoint_dir_path, f'best_decoder_epoch_{epoch}.pth'))
        


def group_weight(module):
    """
        Function for grouping weights and biases of a network into individual groups for training
    """
    group_decay = []
    group_no_decay = []

    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.modules.conv._ConvNd)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    return [
        dict(params=group_decay),
        dict(params=group_no_decay, weight_decay=0),
    ]


def create_optimizers(nets, optim_parameters):
    """
        Creating individual optimizers for encoder and decoder
    """
    (net_encoder, net_decoder, crit) = nets
    
    optimizer_encoder = torch.optim.SGD(group_weight(net_encoder),
                                        lr=optim_parameters["LEARNING_RATE"],
                                        momentum=optim_parameters["MOMENTUM"],
                                        weight_decay=optim_parameters["WEIGHT_DECAY"])
    
    optimizer_decoder = torch.optim.SGD(group_weight(net_decoder),
                                        lr=optim_parameters["LEARNING_RATE"],
                                        momentum=optim_parameters["MOMENTUM"],
                                        weight_decay=optim_parameters["WEIGHT_DECAY"])
        
    return optimizer_encoder, optimizer_decoder


def adjust_learning_rate(optimizers, cur_iter):
    """
        Adjusting learning rate in each iteration
    """
    scale_running_lr = ((1 - cur_iter/TOTAL_NUM_ITER) ** 0.9)
    start_lr = OPTIMIZER_PARAMETERS["LEARNING_RATE"]

    lr_encoder = start_lr * scale_running_lr
    lr_decoder = start_lr * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = lr_decoder


# def stupid_collate_fn(x):
#     return x
#
# def main_train(ckpt_dir_path,
#                data_root_path,
#                continue_training=False,
#                path_encoder_weights="",
#                path_decoder_weights=""):
#     """
#         Main function for training original encoder/decoder architecture for semantic segmentation of 150 classes,
#         with an option to start from pretrained model, trained in last_epoch_trained
#         TODO: Change the name of this function, to be more clear what each of these train functions does
#     """
#     # Encoder/Decoder weights
#     if continue_training:
#         last_epoch = [int(x.split('.')[0].split('_')[-1]) for x in os.listdir(ckpt_dir_path) if x.startswith('encoder_epoch_')][0]
#         path_encoder_weights = os.path.join(ckpt_dir_path, f'encoder_epoch_{last_epoch}.pth')
#         path_decoder_weights = os.path.join(ckpt_dir_path, f'decoder_epoch_{last_epoch}.pth')
#         print(f"The training will continue from {last_epoch + 1} epoch...")
#     else:
#         last_epoch = 0
#         if os.path.exists(ckpt_dir_path):
#             shutil.rmtree(ckpt_dir_path)
#         os.mkdir(ckpt_dir_path)
#
#     net_encoder = build_encoder(path_encoder_weights)
#     net_decoder = build_decoder(path_decoder_weights, use_softmax=False)
#
#     # Creating criterion. In the dataset there are labels -1 which stand for "don't care", so should be ommited during training.
#     crit = nn.NLLLoss(ignore_index=-1)
#
#     # Creating Segmentation Module
#     segmentation_module = SegmentationModule(net_encoder, net_decoder).to(DEVICE)
#
#     # Dataset and Loader
#     dataset_train = TrainDataset(data_root_path, ODGT_TRAINING, batch_per_gpu=BATCH_PER_GPU) # TODO check TrainDataset, change batch_per_gpu to be from constants.py
#
#     loader_train = torch.utils.data.DataLoader(dataset_train,
#                                                batch_size=1, # TODO: write why it is one (because the batch is created in TrainDataset)
#                                                shuffle=False,
#                                                collate_fn=stupid_collate_fn,
#                                                num_workers=NUM_WORKERS, #TODO change to parameter from config.json/contants.py
#                                                drop_last=True,
#                                                pin_memory=True)
#
#     # create loader iterator
#     iterator_train = iter(loader_train)
#
#     # Set up optimizers
#     nets = (net_encoder, net_decoder, crit)
#     optimizers = create_optimizers(nets, OPTIMIZER_PARAMETERS)
#
#     # Tensorboard initialization
#     dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#     writer = SummaryWriter(os.path.join(ckpt_dir_path, 'tensorboard'))
#
#     print('Starting training')
#     # Main loop of certain number of epochs
#     for epoch in range(last_epoch, NUM_EPOCHS):
#         print(f'Training epoch {epoch + 1}/{NUM_EPOCHS}...')
#         train_one_epoch(segmentation_module, iterator_train, optimizers, epoch+1, crit, writer)
#         checkpoint(nets, epoch + 1, ckpt_dir_path, False) #TODO add validation step
#
#     writer.close()
#     print('Training Done!')

"""
def main_train_wall(DEVICE, train_all=False, train_decoder_all=False, start_lr=0.02, last_epoch_trained=0):
    ""
        Main function for training encoder/decoder architecture for semantic segmentation of only wall
        TODO: Change the name of this function, to be more clear what each of these train functions does
    ""
    
    if last_epoch_trained > 0: # When training starts from the previous version of the model
        net_encoder = build_encoder(pretrained=True, epoch=last_epoch_trained, train_only_wall=True)
        net_decoder = build_decoder(pretrained=True, epoch=last_epoch_trained,
                                     use_softmax=False, train_only_wall=True)
    else:
        if train_all:
            # Creates an encoder and a decoder where all parameters are trained
            net_encoder = build_encoder(pretrained=False)
            net_decoder = build_decoder(pretrained=False, use_softmax=False)
        else:
            # Loads a pretrained model of an encoder
            net_encoder = build_encoder(pretrained=True, epoch=NUM_EPOCHS)
            for param in net_encoder.parameters():
                param.requires_grad = False

            if train_decoder_all:
                # Creates a decoder where all parameters are trained
                net_decoder = build_decoder(pretrained=False, use_softmax=False)
            else:
                # Loads a pretrained decoder
                net_decoder = build_decoder(pretrained=True, epoch=NUM_EPOCHS, use_softmax=False)
                for param in net_decoder.parameters():
                    param.requires_grad = False
                    
        # Last layer of the decoder is trained
        net_decoder.conv_last[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    
    # Creating Segmentation Module
    segmentation_module = SegmentationModule(net_encoder, net_decoder).to(DEVICE)
    
    # Dataset and Loader
    dataset_train = TrainDataset("data/", "data/training.odgt", batch_per_gpu=2, train_only_wall=True)
    
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=16,
        drop_last=True,
        pin_memory=True)
    
    # Creating criterion. In the dataset there are labels -1 which stand for "don't care/background", so should be ommited during training
    crit = nn.NLLLoss(ignore_index=-1, weight=torch.tensor([1, 1], dtype=torch.float).to(DEVICE))
    
    # create loader iterator
    iterator_train = iter(loader_train)
    
    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, OPTIMIZER_PARAMETERS)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}
        
    # Main loop of certain number of epochs
    print('Starting wall training:')
    for epoch in range(last_epoch_trained, NUM_EPOCHS):
        train_one_epoch(DEVICE, segmentation_module, iterator_train, optimizers, history, epoch+1, crit, start_lr=start_lr)
        checkpoint(nets, history, epoch+1, train_only_wall=True)

    print('Wall Learning training Done!')"""