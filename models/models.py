import torch
import torch.nn as nn
from models.resnet import resnet50
from functools import partial
from utils.constants import DEVICE, FC_DIM, NUM_CLASSES


class SegmentationModule(nn.Module):
    """
        Segmentation Module class
    """
    def __init__(self, net_encoder, net_decoder):
        super(SegmentationModule, self).__init__()
        self.encoder = net_encoder
        self.decoder = net_decoder
        
    def forward(self, input_dict, seg_size=None):
        """
        Forward pass of Segmentation Module
        """
    
        return self.decoder(self.encoder(input_dict['img_data'].to(DEVICE)), seg_size=seg_size)

   
def build_encoder(path_encoder_weights=""):
    """
        Function for building the encoder part of the Segmentation Module
    """
    pretrained = path_encoder_weights != ""
    orig_resnet = resnet50(pretrained=not pretrained)
    net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)

    if pretrained:
        print('Loading weights for net_encoder')
        net_encoder.load_state_dict(
            torch.load(path_encoder_weights, map_location=lambda storage, loc: storage), strict=False)

    return net_encoder


def build_decoder(path_decoder_weights=""):
    """
        Function for building the decoder part of the Segmentation Module
    """
    net_decoder = PPM(num_class=NUM_CLASSES, fc_dim=FC_DIM)
    
    pretrained = path_decoder_weights != ""
    if pretrained:        
        print('Loading weights for net_decoder')
        net_decoder.load_state_dict(            
            torch.load(path_decoder_weights, map_location=lambda storage, loc: storage), strict=False)
    else:
        net_decoder.apply(weights_init) 
    
    return net_decoder


def weights_init(m):
    """
        Function for initializing weights of the given network
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)


class ResnetDilated(nn.Module):
    """
        Dilated ResNet class, created by dilating original ResNet architecture
    """
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))

        # take pretrained ResNet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        
        self.maxpool = orig_resnet.maxpool
        
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
    
    def _nostride_dilate(self, m, dilate):
        """
            Function for dilation of ResNet
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            elif m.kernel_size == (3, 3):  # other convolutions
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)
    
    def forward(self, x):
        """
            Forward pass of the Dilated ResNet architecture
        """    
        x = nn.Sequential(self.conv1, self.bn1, self.relu1,
                          self.conv2, self.bn2, self.relu2,
                          self.conv3, self.bn3, self.relu3,
                          self.maxpool,
                          self.layer1, self.layer2, self.layer3, self.layer4)(x)
        return x


class PPM(nn.Module):
    """
        Pyramid Pooling Module (PPM) class
    """
    def __init__(self, num_class, fc_dim, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale),
                                          nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(512),
                                          nn.ReLU(inplace=True)))
            
        self.ppm = nn.ModuleList(self.ppm)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales)*512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
    
    def forward(self, x, seg_size=None):
        """
            Forward pass of the PPM architecture
        """
        input_size = x.size()
        ppm_out = [x]
        
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(x),
                                                     (input_size[2], input_size[3]),
                                                     mode='bilinear',
                                                     align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if seg_size:  # is True during inference
            x = nn.functional.interpolate(x, size=seg_size, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:        
            x = nn.functional.log_softmax(x, dim=1)
        
        return x
