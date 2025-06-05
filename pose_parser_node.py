import os
import torch
import torch.nn.functional as F # Still useful for general tensor ops if needed
import cv2
import numpy as np
from PIL import Image
import folder_paths # ComfyUI utility for paths

# --- Hugging Face Transformers Imports ---
from transformers import pipeline

# --- Global Model Initializers (Load once) ---
SEGFORMER_PIPELINE = None

# --- Category Mappings ---
CATEGORY_MAPPINGS = {
    "bottom": {
        "segformer": ['Pants', 'Skirt'],
        "densepose_parts": [7, 9, 8, 10, 11, 13, 12, 14] # Part IDs to extract from input DensePose map
    },
    "person_outline_from_densepose": { # Example: just use densepose for a general person mask
        "segformer": [], # No segformer for this specific category
        "densepose_parts": list(range(1, 25)) # All valid DensePose parts
    },
    "upper_body_clothing": {
        "segformer": ['Upper-clothes', 'Coat', 'Jacket', 'Shirt', 'Sweater', 'T-shirt'],
        "densepose_parts": [1, 2, 3, 4, 5, 6] # Torso, Head, Upper Arms R&L (adjust as needed)
    }
    # Add other categories as needed
}
AVAILABLE_CATEGORIES = list(CATEGORY_MAPPINGS.keys())

# --- Utility functions for ComfyUI image conversion ---
def tensor_to_pil(tensor: torch.Tensor, batch_index=0):
    """Converts a ComfyUI image tensor (BCHW, float 0-1) to a PIL Image (RGB)."""
    if tensor.ndim == 4:
        tensor = tensor[batch_index]
    # Permute CHW to HWC, scale, convert to numpy, then to PIL
    image_np = tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    if image_np.shape[2] == 1: # Grayscale tensor
        return Image.fromarray(image_np.squeeze(), 'L').convert('RGB')
    return Image.fromarray(image_np, 'RGB')

def np_mask_to_tensor(mask_np_bool: np.ndarray):
    """Converts a boolean NumPy mask (HW) to a ComfyUI MASK tensor (B1HW, float 0-1)."""
    mask_float = mask_np_bool.astype(np.float32)
    tensor = torch.from_numpy(mask_float).unsqueeze(0).unsqueeze(0) # Add Batch and Channel
    return tensor

# --- Human Parsing (Segformer) Function (Adapted) ---
def initialize_segformer_pipeline():
    global SEGFORMER_PIPELINE
    if SEGFORMER_PIPELINE is None:
        try:
            print("Initializing Segformer pipeline (mattmdjaga/segformer_b2_clothes)...")
            device_id = 0 if torch.cuda.is_available() else -1
            SEGFORMER_PIPELINE = pipeline(model="mattmdjaga/segformer_b2_clothes", task="image-segmentation", device=device_id)
            print("Segformer pipeline initialized.")
        except Exception as e:
            print(f"Error initializing Segformer pipeline: {e}")
            SEGFORMER_PIPELINE = "failed_to_init"
    return SEGFORMER_PIPELINE != "failed_to_init"

def generate_parsing_mask_node(img_pil: Image.Image, garment_labels: list) -> np.ndarray:
    if SEGFORMER_PIPELINE is None or SEGFORMER_PIPELINE == "failed_to_init":
        print("Segformer pipeline not ready, returning empty mask.")
        return np.zeros((img_pil.height, img_pil.width), dtype=bool)

    try:
        segments = SEGFORMER_PIPELINE(img_pil)
    except Exception as e:
        print(f"Error during Segformer inference: {e}")
        return np.zeros((img_pil.height, img_pil.width), dtype=bool)
        
    mask_list = []
    original_size_wh = img_pil.size
    for s in segments:
        if s['label'] in garment_labels:
            mask = s['mask']
            if mask.size != original_size_wh:
                mask = mask.resize(original_size_wh, Image.NEAREST)
            mask_list.append(np.array(mask).astype(bool))

    if not mask_list:
        return np.zeros((original_size_wh[1], original_size_wh[0]), dtype=bool)
    return np.logical_or.reduce(mask_list)

# --- DensePose Map Processing Function ---
def process_input_densepose_map(densepose_map_tensor: torch.Tensor, target_part_ids: list, original_H, original_W) -> np.ndarray:
    """
    Processes an input DensePose map (assumed to be a part index map or IUV)
    to extract a binary mask for target_part_ids.
    This function NEEDS to be adapted based on the actual format of Fannovel16's node output.
    """
    print("Processing input DensePose map...")
    if densepose_map_tensor is None or not target_part_ids:
        return np.zeros((original_H, original_W), dtype=bool)

    # --- THIS IS THE CRITICAL PART TO ADAPT ---
    # Assumption 1: densepose_map_tensor is a single-channel (grayscale) tensor where pixel values are part indices (0-24).
    # This is a common output format for DensePose part segmentation.
    try:
        if densepose_map_tensor.ndim == 4: # BCHW
            densepose_map_tensor = densepose_map_tensor[0] # Get first image in batch
        if densepose_map_tensor.shape[0] == 3: # If it's an RGB IUV map, convert to something usable
             # This would require specific IUV colormap knowledge to map colors to part IDs.
             # For simplicity, let's assume it's a part index map for now.
             # If it's IUV, you need to convert it to a part index map first or directly extract from IUV.
            print("Warning: Input DensePose map appears to be RGB (possibly IUV). Current logic expects a part index map. Mask might be incorrect.")
            # Convert to grayscale as a naive attempt if it's an IUV-like image
            pil_dp_map = tensor_to_pil(densepose_map_tensor.unsqueeze(0)) # Add batch dim back for tensor_to_pil
            dp_map_np = np.array(pil_dp_map.convert('L'))
        elif densepose_map_tensor.shape[0] == 1: # Grayscale (B1HW)
            dp_map_np = densepose_map_tensor.squeeze().cpu().numpy() # Remove Channel and Batch, convert to NumPy
            dp_map_np = (dp_map_np * 255).astype(np.uint8) # If it was 0-1 float, scale to 0-255 int
        else:
            print(f"Unexpected DensePose map tensor shape: {densepose_map_tensor.shape}. Cannot process.")
            return np.zeros((original_H, original_W), dtype=bool)


        # Resize dp_map_np to original image dimensions if necessary
        if dp_map_np.shape[0] != original_H or dp_map_np.shape[1] != original_W:
            print(f"Resizing DensePose map from {dp_map_np.shape} to ({original_H}, {original_W})")
            dp_map_np = cv2.resize(dp_map_np, (original_W, original_H), interpolation=cv2.INTER_NEAREST)
        
        combined_dp_mask_bool = np.zeros((original_H, original_W), dtype=bool)
        for part_id in target_part_ids:
            combined_dp_mask_bool[dp_map_np == part_id] = True
        
        print(f"DensePose mask generated for parts {target_part_ids} from input map.")
        return combined_dp_mask_bool

    except Exception as e:
        print(f"Error processing input DensePose map: {e}")
        return np.zeros((original_H, original_W), dtype=bool)
    # --- END OF CRITICAL ADAPTATION PART ---


class ComfyCombinedSegmenterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), # Original image for Segformer and shape ref
                "densepose_map_input": ("IMAGE",), # Output from Fannovel16's DensePose Node (or similar)
                "category": (AVAILABLE_CATEGORIES, {"default": AVAILABLE_CATEGORIES[0] if AVAILABLE_CATEGORIES else "bottom"}),
                # prob_threshold might not be directly applicable if the input densepose_map is already binary or part indices
                "closing_kernel_size": ("INT", {"default": 0, "min": 0, "max": 51, "step": 1}),
                "dilation_kernel_size": ("INT", {"default": 0, "min": 0, "max": 51, "step": 1}),
                "apply_convex_hull": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute_segmenter"
    CATEGORY = "Image/Segment" # Or your custom category

    def execute_segmenter(self, image: torch.Tensor, densepose_map_input: torch.Tensor, category: str,
                          closing_kernel_size: int, dilation_kernel_size: int, apply_convex_hull: bool):

        if not initialize_segformer_pipeline():
            print("Node: Segformer pipeline failed to initialize. Segformer part will be skipped.")
            # Fallback: create an empty mask or handle error appropriately
            # For now, let's proceed, parsing_mask_bool will be empty.

        img_pil_rgb = tensor_to_pil(image)
        original_H, original_W = img_pil_rgb.height, img_pil_rgb.width

        selected_map = CATEGORY_MAPPINGS.get(category, {})
        seg_labels = selected_map.get("segformer", [])
        dp_ids_for_category = selected_map.get("densepose_parts", [])

        # --- Process Input DensePose Map ---
        smoother_dp_mask_bool = np.zeros((original_H, original_W), dtype=bool)
        if dp_ids_for_category: # Only process if the category uses densepose parts
            smoother_dp_mask_bool = process_input_densepose_map(densepose_map_input, dp_ids_for_category, original_H, original_W)
        else:
            print("Node: No DensePose parts defined for this category or skipping DensePose.")


        # --- Generate Segformer Parsing Mask ---
        parsing_mask_bool = np.zeros((original_H, original_W), dtype=bool)
        if seg_labels and (SEGFORMER_PIPELINE and SEGFORMER_PIPELINE != "failed_to_init"):
            print("Node: Generating Segformer parsing mask...")
            parsing_mask_bool = generate_parsing_mask_node(img_pil_rgb, seg_labels)
        else:
            print("Node: Skipping Segformer (no labels for category or pipeline issue).")

        # --- Combine Masks ---
        # Ensure shapes match before combining (they should be original_H, original_W now)
        if smoother_dp_mask_bool.shape != (original_H, original_W):
            smoother_dp_mask_bool = cv2.resize(smoother_dp_mask_bool.astype(np.uint8), (original_W, original_H), interpolation=cv2.INTER_NEAREST).astype(bool)
        if parsing_mask_bool.shape != (original_H, original_W):
            parsing_mask_bool = cv2.resize(parsing_mask_bool.astype(np.uint8), (original_W, original_H), interpolation=cv2.INTER_NEAREST).astype(bool)
            
        processed_mask_bool = np.logical_or(smoother_dp_mask_bool, parsing_mask_bool)
        processed_mask_uint8 = processed_mask_bool.astype(np.uint8) * 255

        # --- Apply Convex Hull ---
        if apply_convex_hull and np.any(processed_mask_uint8): # Only apply if mask is not empty
            print("Node: Applying convex hull...")
            contours, _ = cv2.findContours(processed_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_contour_points = np.concatenate(contours)
                if len(all_contour_points) >= 3:
                    convex_hull_of_points = cv2.convexHull(all_contour_points)
                    filled_hull_mask = np.zeros_like(processed_mask_uint8)
                    cv2.drawContours(filled_hull_mask, [convex_hull_of_points], 0, 255, cv2.FILLED)
                    processed_mask_uint8 = filled_hull_mask
                else: print("Node: Not enough contour points for convex hull.")
            else: print("Node: No contours found for convex hull.")

        # --- Morphological Operations ---
        if closing_kernel_size > 0 and np.any(processed_mask_uint8):
            eff_close_k = closing_kernel_size + (1 - closing_kernel_size % 2) # Ensure odd
            if eff_close_k > 0 : # Make sure it's positive
                print(f"Node: Applying closing with kernel {eff_close_k}x{eff_close_k}...")
                closing_kernel = np.ones((eff_close_k, eff_close_k), np.uint8)
                processed_mask_uint8 = cv2.morphologyEx(processed_mask_uint8, cv2.MORPH_CLOSE, closing_kernel)

        if dilation_kernel_size > 0 and np.any(processed_mask_uint8):
            eff_dilate_k = dilation_kernel_size + (1 - dilation_kernel_size % 2) # Ensure odd
            if eff_dilate_k > 0:
                print(f"Node: Applying dilation with kernel {eff_dilate_k}x{eff_dilate_k}...")
                dilation_kernel = np.ones((eff_dilate_k, eff_dilate_k), np.uint8)
                processed_mask_uint8 = cv2.dilate(processed_mask_uint8, dilation_kernel, iterations=1)
        
        final_mask_tensor = np_mask_to_tensor(processed_mask_uint8.astype(bool))
        return (final_mask_tensor,)


NODE_CLASS_MAPPINGS = {
    "ComfyCombinedSegmenter": ComfyCombinedSegmenterNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyCombinedSegmenter": "Combined Segmenter (DP Input + Segformer)"
}
