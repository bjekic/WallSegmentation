import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_PER_GPU = 2
NUM_WORKERS = 6

TRAIN_SUBSAMPLE_DATASET = True

NUM_ITER_PER_EPOCH = 5000
NUM_EPOCHS = 20
TOTAL_NUM_ITER = NUM_ITER_PER_EPOCH * NUM_EPOCHS

IMG_SIZES = (300, 375, 450, 525, 575)
IMG_MAX_SIZE = 900
PADDING = 8  # 8 when dilated, 32 when not dilated
SEGM_DOWNSAMPLING_RATE = 8  # 8 when dilated, 32 when not dilated
LIST_SCENES = ['bathroom', 'bedroom', 'kitchen', 'living_room', 'art_gallery', 'art_studio', 'attic', 'auditorium',
               'shop', 'ballroom', 'bank_indoor', 'banquet_hall', 'bar', 'basement', 'bookstore', 'childs_room',
               'classroom', 'room', 'closet', 'clothing_store', 'computer_room', 'conference_room', 'corridor',
               'office', 'darkroom', 'dentists_office', 'diner_indoor', 'dinette_home', 'dining_room', 'doorway_indoor',
               'dorm_room', 'dressing_room', 'entrance_hall', 'galley', 'game_room', 'garage_indoor', 'gymnasium_indoor',
               'hallway', 'home_office', 'hospital_room', 'hotel_room', 'jail_cell', 'kindergarden_classroom',
               'lecture_room', 'library_indoor', 'lobby', 'museum_indoor', 'nursery', 'playroom', 'staircase',
               'television_studio', 'utility_room', 'waiting_room', 'warehouse_indoor', 'youth_hostel']

OPTIMIZER_PARAMETERS = {
        "LEARNING_RATE": 0.02,
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 1e-4
    }

NUM_CLASSES = 2
FC_DIM = 2048  # 512 for resnet18, for rest it is 2048

ODGT_TRAINING = "data/training.odgt"
ODGT_EVALUTATION = "data/validation.odgt"

SCENE_CATEGORIES = 'ADEChallengeData2016/sceneCategories.txt'