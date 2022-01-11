import torch
import torch.nn as nn
from . import resnet 
from functools import partial


class SegmentationModule(nn.Module):
    """
        Segmentation Module class
    """
    def __init__(self, net_enc, net_dec):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        
    def forward(self, feed_dict, segSize=None):
        """
        Forward pass of Segmentation Module
        """
    
        return self.decoder(
            self.encoder(feed_dict['img_data'].to('cuda')), segSize=segSize
        )

   
def build_encoder(pretrained=True, epoch=20, train_only_wall=False):
    """
        Function for building the encoder part of the Segmentation Module
    """
    orig_resnet = resnet.resnet50(pretrained=not pretrained)
    net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
    
    if train_only_wall:        
        weights = 'ckpt/wall_encoder_epoch_' + str(epoch) + '.pth'
    else:
        weights = 'ckpt/encoder_epoch_' + str(epoch) + '.pth'
    
    if pretrained:
        print('Loading weights for net_encoder')
        net_encoder.load_state_dict(
            torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        
    return net_encoder


def build_decoder(pretrained=True, epoch=20, fc_dim=2048, num_class=150, use_softmax=True,
                   train_only_wall=False):
    """
        Function for building the decoder part of the Segmentation Module
    """ 
    net_decoder = PPM(        
        num_class=num_class,
        fc_dim=fc_dim,
        use_softmax=use_softmax)
    
    net_decoder.apply(weights_init)
    
    # When flag "train_only_wall" is set to true, the last layer of decoder is set to have only 2 classes
    if train_only_wall: 
        net_decoder.conv_last[4] = torch.nn.Conv2d(512, 2, kernel_size=1)
        weights = 'ckpt/wall_decoder_epoch_' + str(epoch) + '.pth'
    else:
        weights = 'ckpt/decoder_epoch_' + str(epoch) + '.pth'
    
    if pretrained:        
        print('Loading weights for net_decoder')
        net_decoder.load_state_dict(            
            torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
    
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
            orig_resnet.layer4.apply(partial( self._nostride_dilate, dilate=4))

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
            elif m.kernel_size == (3, 3): # other convolutions
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
                        self.layer1, self.layer2, self.layer3, self.layer4) (x)        
        return x


class PPM(nn.Module):
    """
        Pyramid Pooling Module (PPM) class
    """
    def __init__(self, num_class=150, fc_dim=4096, use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ) )
            
        self.ppm = nn.ModuleList(self.ppm)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales)*512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
    
    def forward(self, x, segSize=None):
        """
            Forward pass of the PPM architecture
        """
        input_size = x.size()
        ppm_out = [x]
        
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(x), (input_size[2], input_size[3]),
                                                      mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:        
            x = nn.functional.log_softmax(x, dim=1)
        
        return x
