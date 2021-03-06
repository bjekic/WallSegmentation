import os, json, torch
from torchvision import transforms
import numpy as np
from PIL import Image

# Function for image resizing
def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

# Function for creating a dictionary where keys are image names, and values are scene in the image
def create_scene_dict(path, list_scenes):
    dict_scene = {}
    
    file = open(path,'r')
    counter_val = 0
    counter_train = 0
    for line in file:
        temp = line.split(' ')
        scene = temp[1][:-1]
        dict_scene[temp[0]] = scene
        if scene in list_scenes and temp[0][:7]=='ADE_val':
            counter_val +=1
        if scene in list_scenes and temp[0][:9]=='ADE_train':
            counter_train +=1
    return dict_scene, counter_val, counter_train

# Base class for dataset
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, **kwargs):
        # parse options
        self.imgSizes = (300, 375, 450, 525, 600)
        self.imgMaxSize = 1000
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = 8

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        
        self.list_scenes = ['bathroom','bedroom','kitchen','living_room','art_gallery','art_studio','attic','auditorium',\
                           'shop','ballroom','bank_indoor','banquet_hall','bar','basement','bookstore','childs_room',\
                           'classroom','room','closet','clothing_store','computer_room','conference_room','corridor',\
                           'office','darkroom','dentists_office','diner_indoor','dinette_home','dining_room','doorway_indoor',\
                           'dorm_room','dressing_room','entrance_hall','galley','game_room','garage_indoor','gymnasium_indoor',\
                           'hallway','home_office','hospital_room','hotel_room','jail_cell','kindergarden_classroom',\
                           'lecture_room','library_indoor','lobby','museum_indoor','nursery','playroom','staircase',\
                           'television_studio','utility_room','waiting_room','warehouse_indoor','youth_hostel']
        
        
    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

# Dataset for training images that inherits from the base class
class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, batch_per_gpu=1, train_only_wall = False, **kwargs):
        super(TrainDataset, self).__init__(odgt, **kwargs)
        # Added attribute:
        self.train_only_wall = train_only_wall # flag that indicates whether the whole database is used or only a part
        
        self.root_dataset = root_dataset
        self.num_sample = len(self.list_sample)
        
        # down sampling rate of segm labe
        self.segm_downsampling_rate = 8
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False
        
        # Added attribute:
        self.scene_dict, _ , num_ex_train= create_scene_dict( self.root_dataset+'ADEChallengeData2016/sceneCategories.txt', self.list_scenes )
        
        if self.train_only_wall:
            print( 'Number of different images: ' + str(num_ex_train) )
        else:
            print( 'Number of different images: 20210')
            
    
    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            
            # update current sample pointer
            self.cur_idx += 1
            
            # If only a subpart of the database is used, check whether the current image has the appropriate scene. If not, continue while loop
            if self.train_only_wall:
                scene = self.scene_dict[ this_sample['fpath_img'][37:55] ] # gets the scene of the particular image
                if not( scene in self.list_scenes ):
                    continue
            
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            
            if self.cur_idx >= self.num_sample:
                self.ccccur_idx = 0
                np.random.shuffle(self.list_sample)

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
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or larger than segm downsampling rate'
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            img = Image.open(image_path).convert('RGB')
            segm = Image.open(segm_path)
            assert(segm.mode == "L")
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])
            
            # random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

            # further downsample seg label, need to avoid seg label misalignment
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(
                segm_rounded,
                (segm_rounded.size[0] // self.segm_downsampling_rate, \
                 segm_rounded.size[1] // self.segm_downsampling_rate), \
                interp='nearest')

            # image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)

            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)
            
            # Check if training only wall, there is no need for additional labels, except 0 and 1 (where 0 represents wall and 1 represents other)
            if self.train_only_wall:
                segm[segm>0] = 1
                
            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        
        return output

    def __len__(self):
        return int(1e6) # It's a fake length due to the trick that every loader maintains its own list
   
# Dataset for validation images that inherits from the base class 
# Validation dataset only for images of interest (images containing wall)
class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, **kwargs):
        super(ValDataset, self).__init__(odgt, **kwargs)
        self.root_dataset = root_dataset
        self.scene_dict, self.num_sample, _ = create_scene_dict( self.root_dataset + 'ADEChallengeData2016/sceneCategories.txt', self.list_scenes )
        self.index = 0
        
    def __getitem__(self, index):        
        while True:   
            this_record = self.list_sample[self.index]
            scene = self.scene_dict[ this_record['fpath_img'][39:55] ] # gets the scene of the particular image
            self.index += 1
            if scene in self.list_scenes:
                break
        
        # load image and label
        image_path = os.path.join( self.root_dataset, this_record['fpath_img'] )
        segm_path = os.path.join( self.root_dataset, this_record['fpath_segm'] )
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        
        img = self.img_transform(img)
        segm = self.segm_transform(segm)
        
        segm[segm>0] = 1
        
        output = dict()
        
        output['img_data'] = img[None]
        output['seg_label'] = segm       
        output['name'] = this_record['fpath_img'][39:55]
        
        return output

    def __len__(self):
        return self.num_sample    

        
        
        