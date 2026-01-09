import torch


IMG_SIZE = 256

DATA_SET_DIR = "./data/orig_inpainted"
PREDICTION_DIR = "./test_predictions" # dir pro výsledky
BATCH_SIZE = 8
EPOCHS = 120
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_SPLIT = 0.7  
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15