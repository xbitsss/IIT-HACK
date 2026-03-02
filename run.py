import torch
import numpy as np
import os
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Path to the Weights you just downloaded
checkpoint = "./sam2_hiera_large.pt"

# 3. Path to the Config (This is inside your cloned folder)
# Adjust the 'segment-anything-2' part if your folder has a different name
model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml" 

# 4. Initialize Model
try:
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # If the above fails, it's usually because the 'model_cfg' path is wrong.
    exit()

# 5. Load Image
image_path = os.path.join("images", "f2.jpg")
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# 6. Predict
predictor.set_image(image_np)

# We use numpy arrays for the coordinates
input_point = np.array([[500, 375]])
input_label = np.array([1])

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# 7. Result
best_mask = masks[scores.argmax()]
print("SUCCESS! Mask shape:", best_mask.shape)