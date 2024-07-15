import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Predict classes and hyperparameters for GroundingDINO
CLASSES = ["central part of the scale pattern on the chest without head and arms"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.7