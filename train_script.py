from train import main_train
import json

with open('configs/config_desktop_everseen.json', 'r') as f:
    config = json.load(f)

main_train(ckpt_dir_path=config["CHECKPOINT_DIR_PATH"],
           data_root_path=config["ROOT_DATASET"],
           continue_training=config["CONTINUE_TRAINING"],
           path_encoder_weights=config["MODEL_ENCODER_WEIGHTS_PATH"],
           path_decoder_weights=config["MODEL_DECODER_WEIGHTS_PATH"]
           )