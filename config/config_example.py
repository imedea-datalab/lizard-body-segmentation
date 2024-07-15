# Images and labels
DATA_FOLDER = "your/data/folder"
ROBOFLOW_API_KEY = "your_roboflow_api_key"

# Input Data
RAW_IMAGES_PATH = f"{DATA_FOLDER}/FotosLizards"

# Output Data
SEGMENTED_LIZARDS_PATH = f"{DATA_FOLDER}/segmented-lizards"
SEGMENTED_IMAGES_PATH = f"{SEGMENTED_LIZARDS_PATH}/images"
SEGMENTED_LABELS_PATH = f"{SEGMENTED_LIZARDS_PATH}/labels"

# Here we will store the images and labels that we will use to train the model
SELECTED_IMAGES_PATH = f"{SEGMENTED_LIZARDS_PATH}/selected"
IMAGES_PATH = f"{SELECTED_IMAGES_PATH}/images"
LABELS_PATH = f"{SELECTED_IMAGES_PATH}/labels"

# Here we will store all the segmented images after applying the model
PROCESSED_IMAGES_PATH = f"{DATA_FOLDER}/FotosLizards-segmented"


# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "model_utils/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "model_utils/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "model_utils/sam_vit_h_4b8939.pth"

# Best segmentation model
YOLO_SEGMENTATION_MODEL = "runs/segment/train7/weights/best.pt"
