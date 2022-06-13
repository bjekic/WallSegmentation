import os, json, torch
from torchvision import transforms
import numpy as np
from PIL import Image
from utils.constants import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZES, IMG_MAX_SIZE, PADDING, \
                            LIST_SCENES, TRAIN_SUBSAMPLE_DATASET, SCENE_CATEGORIES, SEGM_DOWNSAMPLING_RATE
from utils.utils import imresize


def create_scene_dict(path, list_scenes):
    """
        Function for creating a dictionary where keys are image names, and values are scene in the image
    """
    dict_scene = {}
    
    file = open(path, 'r')
    counter_val = 0
    counter_train = 0
    for line in file:
        temp = line.split(' ')
        scene = temp[1][:-1]
        dict_scene[temp[0]] = scene
        if scene in list_scenes and temp[0].startswith('ADE_val'):
            counter_val += 1
        if scene in list_scenes and temp[0].startswith('ADE_train'):
            counter_train += 1
    return dict_scene, counter_val, counter_train


class BaseDataset(torch.utils.data.Dataset):
    """
        Base custom dataset class
    """
    def __init__(self, odgt, **kwargs):
        self.imgSizes = IMG_SIZES
        self.imgMaxSize = IMG_MAX_SIZE
        
        self.padding_constant = PADDING
        self.list_scenes = LIST_SCENES

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD) 
        
        
    def parse_input_list(self, odgt):
        """
            Function for parsing input list
        """
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]


    def img_transform(self, img):
        """
            Function for image transform (re-order image channels and normalize image)
        """
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        """
            Function for transorming segmentation mask from numpy array to tensor with values in the range [-1, 149]
        """
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    def round2nearest_multiple(self, x, p):
        """ 
            Function for rounding x to the nearest multiple of p and x' >= x
        """
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    """
        Train dataset class
    """
    def __init__(self, root_dataset, odgt, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(odgt, **kwargs)
        
        self.train_subsample_dataset = TRAIN_SUBSAMPLE_DATASET # flag that indicates whether the whole database is used or only a part
        self.root_dataset = root_dataset
        self.num_sample = len(self.list_sample)
        
        # Down sampling rate of segm label
        self.segm_downsampling_rate = SEGM_DOWNSAMPLING_RATE
        self.batch_per_gpu = batch_per_gpu

        # Classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # Override dataset length when training with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False
        
        self.scene_dict, _, num_ex_train = create_scene_dict(os.path.join(self.root_dataset, SCENE_CATEGORIES), self.list_scenes)
        
        if self.train_subsample_dataset:
            print(f'Number of different images: {num_ex_train}')
        else:
            print(f'Number of different images: {self.num_sample}')
            
    
    def _get_sub_batch(self):
        """
        Used to group images with same type of aspect ratio into the same batch
        :return: batch of images which all have either larger width or larger height
        """
        while True:
            # Get a sample record
            this_sample = self.list_sample[self.cur_idx]

            # Update current sample pointer
            self.cur_idx += 1
            
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            # If only a subpart of the database is used, check whether the current image has the appropriate scene. If not, continue while loop.
            if self.train_subsample_dataset:
                this_sample_name = this_sample['fpath_img'].split(".")[0].split(os.path.sep)[-1]
                scene = self.scene_dict[this_sample_name]  # gets the scene of the particular image
                if scene not in self.list_scenes:
                    continue

            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample)  # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample)  # h <= w, go to 2nd class

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break

        return batch_records

    def __getitem__(self, index):
        """
        Obtains batch used for training
        :param index:
        :return: a dictionary with 'img_data' containing batch of images, and 'seg_label' which are segmentation labels
        for images in 'img_data'
        """
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # Get sub-batch candidates
        batch_records = self._get_sub_batch()

        # Resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, (list, tuple)):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # Calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(this_short_size / min(img_height, img_width),
                             self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, 'padding constant must be equal or larger than segm downsampling rate'
        
        batch_images = torch.zeros(self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(self.batch_per_gpu,
                                  batch_height // self.segm_downsampling_rate,
                                  batch_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # Load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            img = Image.open(image_path).convert('RGB')
            segm = Image.open(segm_path)
            assert(segm.mode == "L")
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])

            # Random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

            # Note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

            # Further downsample seg label, need to avoid seg label misalignment
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(segm_rounded,
                            (segm_rounded.size[0] // self.segm_downsampling_rate, 
                            segm_rounded.size[1] // self.segm_downsampling_rate),
                            interp='nearest')

            # Image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)

            # Segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)

            # Check if training only wall, there is no need for additional labels, except 0 and 1 (where 0 represents wall and 1 represents other)
            if self.train_subsample_dataset:
                segm[segm > 0] = 1

            # Put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        return {'img_data': batch_images, 'seg_label': batch_segms}

    def __len__(self):
        return int(1e8)  # It's a fake length due to the trick that every loader maintains its own list
   

class ValDataset(BaseDataset):
    """
        Dataset for validation images that inherits from the base class Validation dataset,
        only for images of interest (images containing wall)
    """
    def __init__(self, root_dataset, odgt, **kwargs):
        super(ValDataset, self).__init__(odgt, **kwargs)
        self.root_dataset = root_dataset
        self.scene_dict, self.num_sample, _ = create_scene_dict(self.root_dataset + SCENE_CATEGORIES, self.list_scenes)
        self.index = 0
        
    def __getitem__(self, index):        
        while True:   
            this_record = self.list_sample[self.index]
            this_record_name = this_record['fpath_img'].split(".")[0].split(os.path.sep)[-1]
            scene = self.scene_dict[this_record_name]  # gets the scene of the particular image
            self.index += 1
            if self.index >= self.num_sample:
                self.index = 0
            if scene in self.list_scenes:
                break

        # Load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)

        img = self.img_transform(img)
        segm = self.segm_transform(segm)

        segm[segm > 0] = 1

        return {
            'img_data': img[None],
            'seg_label': segm,
            'name': this_record_name,
        }

    def __len__(self):
        return self.num_sample    

        
        
        