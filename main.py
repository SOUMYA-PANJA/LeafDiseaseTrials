import cv2
import numpy as np

from ultralytics import YOLO

LEAF_MODEL_PATH = "models/leaf.pt"
LEAF_DISEASE_MODEL_PATH = "models/leafdisease.pt"

def preprocessImage(image: np.ndarray) -> np.ndarray:
    """Preprocess image

    Args:
        image (np.ndarray): Image object of numpy.ndarray type

    Returns:
        np.ndarray: Preprocessed image of numpy.ndarray type
    """
    return cv2.resize(image, (640, 640))

def getSegmentationMask(MODEL_PATH: str, image: np.ndarray) -> np.ndarray:
    """Extract Segmentation Mask

    Args:
        MODEL_PATH (str): Path to the segmentation model
        image (np.ndarray): Image object of numpy.ndarray type

    Returns:
        np.ndarray: Binary Segmentation mask
    """
    
    MODEL = YOLO(MODEL_PATH)
    
    result = MODEL(image)
    
    mask = result[0].masks.data
    
    try:
        binary_image = mask.numpy()
    except:
        binary_image = mask.cpu().numpy()
        
    binary_image = binary_image.astype('uint8')
    
    binary_image = np.bitwise_or.reduce(binary_image, axis = 0)
    
    return binary_image

image = cv2.imread("img3.jpg")
image = preprocessImage(image)
binary_mask_leaf = getSegmentationMask(LEAF_MODEL_PATH, image)
binary_mask_leaf_disease = getSegmentationMask(LEAF_DISEASE_MODEL_PATH, image)

print(np.sum(binary_mask_leaf * binary_mask_leaf_disease) / np.sum(binary_mask_leaf) * 100)


color_mask = np.zeros_like(image)
color_mask[binary_mask_leaf_disease == 1] = [255, 0, 0]  # Green color for the mask

# Overlay the mask on the image
overlay = cv2.addWeighted(image, 1, color_mask, 0.3, 0)
output_path = 'overlay_result_leaf.jpg'
cv2.imwrite(output_path, overlay)