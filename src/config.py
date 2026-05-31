import random
import numpy as np
import torch

CLIENT_NUMBER = 50
CLIENT_LABELED_NUMBER = 2
CLIENT_PRE_LABELED_NUMBER = 200
SERVER_NUMBER = 500
EPOCHS = 200
EARLY_STOP_PATIENCE = 10
TEST_NUMBER = 500
SIMILARITY_MODE = 'cosine'
CLASS = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LR = 1e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 2
TOP_K = 0.1
ROUNDS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False