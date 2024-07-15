import cv2
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
from segment_anything import SamPredictor


import params


def mask_to_polygon(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Reshape the contour to be a 2D array
    reshaped_contour = approx_contour.reshape(-1, 2)

    # Normalize the coordinates
    normalized = reshaped_contour / np.array([mask.shape[1], mask.shape[0]])

    # Flatten the normalized coordinates
    flattened = normalized.flatten()

    return flattened


def annotateImageWithDetections(image, detections, add_masks=True):
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        f"{params.CLASSES[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _, _ in detections
    ]
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )

    if add_masks:
        annotated_image = mask_annotator.annotate(
            scene=image.copy(), detections=detections
        )

    return annotated_image


def crop_and_display_image(image, mask):

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    ymax = np.round((ymax + 3 * ymin) / 4).astype(int)
    xmax += 10
    ymin -= 10
    xmin -= 10

    # Crop the image using the bounding box
    cropped_image = image[ymin:ymax, xmin:xmax]

    # If you want to apply the mask to the cropped image, do this:
    cropped_mask = mask[ymin:ymax, xmin:xmax]
    cropped_image[cropped_mask == False] = 0

    plt.imshow(cropped_image)
    plt.show()


# Prompting SAM with detected boxes
def segment(
    sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def save_polygon_label_as_yolo_txt(label_path, polygon):
    # Format as YOLO segmentation string
    class_id = 0  # Assuming lizard is class 0
    coords = " ".join([f"{x:.6f}" for x in polygon])
    yolo_line = f"{class_id} {coords}"

    # Save labels as txt
    with open(label_path, "w") as f:
        f.write(yolo_line)
