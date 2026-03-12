import sys
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# --- 1. SETTING UP THE ENVIRONMENT ---
current_dir = os.path.abspath(os.getcwd())
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("✅ SAM 2 modules imported.")
except ImportError as e:
    print(f"❌ ImportError: {e}")
    sys.exit(1)

# --- 2. CONFIGURATION ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Running on: {device.upper()} ---")

checkpoint_path = os.path.join(current_dir, "sam2.1_hiera_l.pt")
config_root = os.path.join(current_dir, "sam2", "configs")
# The filename relative to the config_root
config_file = "sam2.1/sam2.1_hiera_l.yaml"

# --- 3. LOAD MODEL (The "Nuclear" Hydra Fix) ---
print(f"Initializing Hydra and loading model...")

try:
    # Reset Hydra to be safe
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # MANUALLY initialize Hydra pointing to your configs folder
    # 'version_base=None' silences the warning you saw earlier
    with initialize_config_dir(config_dir=config_root, version_base=None):
        # Now that Hydra is initialized, build_sam2 should find its config
        sam2_model = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=device
        )
    
    predictor = SAM2ImagePredictor(sam2_model)
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Initialization Error: {e}")
    print("\n--- Diagnostic Check ---")
    full_yaml_path = os.path.join(config_root, config_file)
    print(f"Checking: {full_yaml_path}")
    print(f"Exists: {os.path.exists(full_yaml_path)}")
    sys.exit(1)

# --- 4. LOAD IMAGE ---
image_path = os.path.join(current_dir, "images", "f2.png")
if not os.path.exists(image_path):
    print(f"❌ Error: Image not found at {image_path}")
    sys.exit(1)

image = Image.open(image_path).convert("RGB")
image_np = np.array(image)
predictor.set_image(image_np)

# --- 5. PREDICT ---
input_point = np.array([[500, 375]])
input_label = np.array([1]) 

print("Predicting mask...")
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# --- 6. SAVE RESULT ---
# Select the mask with the highest confidence score
# We use .squeeze() to turn (1, H, W) into (H, W)
best_mask = masks[scores.argmax()].squeeze()

plt.figure(figsize=(10, 10))
plt.imshow(image_np)

# Create a semi-transparent blue overlay for the mask
# Ensure mask is boolean for indexing
if best_mask.dtype != bool:
    best_mask = best_mask > 0

mask_overlay = np.zeros((image_np.shape[0], image_np.shape[1], 4))
mask_overlay[best_mask] = [0, 0, 1, 0.4]  # Blue with 40% opacity
plt.imshow(mask_overlay)

# Add the prompt point as a red star
plt.scatter(input_point[:, 0], input_point[:, 1], color='red', marker='*', s=200, edgecolors='white')
plt.axis('off')

output_path = os.path.join(current_dir, "output_mask.png")
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"⭐ DONE! Result saved to: {output_path}")