import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from glob import glob
import folder_paths # ComfyUI utility for paths

# --- Hugging Face Transformers Imports ---
from transformers import pipeline

# --- Detectron2 & DensePose Imports ---
_DENSEPOSE_AVAILABLE = False
try:
    from detectron2.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor
    from densepose import add_densepose_config
    _DENSEPOSE_AVAILABLE = True
    print("Detectron2 and DensePose loaded successfully.")
except ImportError:
    print("Warning: Detectron2 or DensePose not found. DensePose functionality will be disabled.")

# --- Global Model Initializers (Load once) ---
SEGFORMER_PIPELINE = None
DENSEPOSE_PREDICTOR_CACHE = {} # Cache predictors by config/model path

# --- Category Mappings (can be moved to a config file or made more dynamic) ---
CATEGORY_MAPPINGS = {
    "bottom": {
        "segformer": ['Pants', 'Skirt'],
        "densepose": [7, 9, 8, 10, 11, 13, 12, 14]
    },
    "full_body_person": { # Example: if you want a general person mask
        "segformer": ['Person'], # Assuming your segformer model has 'Person'
        "densepose": list(range(1, 25)) # All DensePose parts
    }
    # Add other categories as needed
}
AVAILABLE_CATEGORIES = list(CATEGORY_MAPPINGS.keys())

# --- Utility functions for ComfyUI image conversion ---
def tensor_to_pil(tensor):
    """Converts a ComfyUI image tensor (BCHW, float 0-1) to a PIL Image (RGB)."""
    if tensor.ndim == 4: # Batch of images
        tensor = tensor[0] # Take the first image if batched
    # Permute CHW to HWC, scale, convert to numpy, then to PIL
    image_np = tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np, 'RGB')

def pil_to_tensor(pil_image):
    """Converts a PIL Image (RGB) to a ComfyUI image tensor (BCHW, float 0-1)."""
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0) # HWC to CHW, add Batch
    return tensor

def np_mask_to_tensor(mask_np_bool):
    """Converts a boolean NumPy mask (HW) to a ComfyUI MASK tensor (B1HW, float 0-1)."""
    mask_float = mask_np_bool.astype(np.float32)
    tensor = torch.from_numpy(mask_float).unsqueeze(0).unsqueeze(0) # Add Batch and Channel
    return tensor

# --- DensePose Functions (Adapted) ---
def setup_densepose_predictor(config_fpath: str, model_fpath: str, min_score: float = 0.7):
    cache_key = (config_fpath, model_fpath, min_score)
    if cache_key in DENSEPOSE_PREDICTOR_CACHE:
        return DENSEPOSE_PREDICTOR_CACHE[cache_key]

    if not _DENSEPOSE_AVAILABLE:
        print("Detectron2/DensePose not available for predictor setup.")
        return None
    try:
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_score
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.freeze()
        predictor = DefaultPredictor(cfg)
        DENSEPOSE_PREDICTOR_CACHE[cache_key] = predictor
        print(f"DensePose predictor initialized for {os.path.basename(model_fpath)}.")
        return predictor
    except Exception as e:
        print(f"Error setting up DensePose predictor: {e}")
        return None

def run_densepose_inference(img_bgr: np.ndarray, dp_predictor):
    if dp_predictor is None: return None
    with torch.no_grad():
        outputs = dp_predictor(img_bgr)["instances"]
    if len(outputs) == 0 or not outputs.has("pred_boxes") or not outputs.has("pred_densepose"):
        return None
    return outputs

def generate_smoother_densepose_binary_mask_node(img_shape: tuple, densepose_outputs, target_part_ids: list, prob_threshold: float = 0.5) -> np.ndarray:
    H, W = img_shape[:2]
    full_image_binary_mask = np.zeros((H, W), dtype=bool)
    if densepose_outputs is None: return full_image_binary_mask

    pred_boxes = densepose_outputs.pred_boxes.tensor.int().cpu().numpy()
    pred_densepose = densepose_outputs.pred_densepose

    for i in range(len(densepose_outputs)):
        x1, y1, x2, y2 = pred_boxes[i]
        w_box, h_box = x2 - x1, y2 - y1
        if w_box <= 0 or h_box <= 0: continue

        instance_dp_obj = pred_densepose[i]
        raw_fine_segm_tensor = None
        if hasattr(instance_dp_obj, 'fine_segm') and instance_dp_obj.fine_segm.numel() > 0:
            raw_fine_segm_tensor = instance_dp_obj.fine_segm
            if len(raw_fine_segm_tensor.shape) == 4: raw_fine_segm_tensor = raw_fine_segm_tensor[0]
        else: continue
        if len(raw_fine_segm_tensor.shape) != 3: continue

        prob_tensor_dp_res = F.softmax(raw_fine_segm_tensor, dim=0)
        combined_prob_map_dp_res = torch.zeros_like(prob_tensor_dp_res[0])
        for part_id in target_part_ids:
            if 0 < part_id < prob_tensor_dp_res.shape[0]:
                combined_prob_map_dp_res = torch.maximum(combined_prob_map_dp_res, prob_tensor_dp_res[part_id])

        upscaled_prob_map_box_res = cv2.resize(
            combined_prob_map_dp_res.cpu().numpy(),
            (w_box, h_box),
            interpolation=cv2.INTER_LINEAR
        )
        instance_binary_mask_box_res = upscaled_prob_map_box_res > prob_threshold
        full_image_binary_mask[y1:y2, x1:x2] = np.logical_or(
            full_image_binary_mask[y1:y2, x1:x2],
            instance_binary_mask_box_res
        )
    return full_image_binary_mask

# --- Human Parsing (Segformer) Function (Adapted) ---
def initialize_segformer_pipeline():
    global SEGFORMER_PIPELINE
    if SEGFORMER_PIPELINE is None:
        try:
            print("Initializing Segformer pipeline (mattmdjaga/segformer_b2_clothes)...")
            device = 0 if torch.cuda.is_available() else -1
            SEGFORMER_PIPELINE = pipeline(model="mattmdjaga/segformer_b2_clothes", task="image-segmentation", device=device)
            print("Segformer pipeline initialized.")
        except Exception as e:
            print(f"Error initializing Segformer pipeline: {e}")
            SEGFORMER_PIPELINE = "failed_to_init"
    return SEGFORMER_PIPELINE != "failed_to_init"

def generate_parsing_mask_node(img_pil: Image.Image, garment_labels: list) -> np.ndarray:
    if SEGFORMER_PIPELINE is None or SEGFORMER_PIPELINE == "failed_to_init":
        return np.zeros((img_pil.height, img_pil.width), dtype=bool)

    segments = SEGFORMER_PIPELINE(img_pil)
    mask_list = []
    original_size_wh = img_pil.size
    for s in segments:
        if s['label'] in garment_labels:
            mask = s['mask']
            if mask.size != original_size_wh:
                mask = mask.resize(original_size_wh, Image.NEAREST)
            mask_list.append(np.array(mask).astype(bool)) # Ensure boolean

    if not mask_list:
        return np.zeros((original_size_wh[1], original_size_wh[0]), dtype=bool)
    return np.logical_or.reduce(mask_list)


class ComfyPoseParserNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Try to find DensePose models - users might place them in ComfyUI/models/densepose/
        densepose_model_dir = os.path.join(folder_paths.models_dir, "densepose")
        densepose_configs = ["None"]
        densepose_models = ["None"]

        if _DENSEPOSE_AVAILABLE and os.path.isdir(densepose_model_dir):
            densepose_configs.extend([f for f in os.listdir(densepose_model_dir) if f.endswith(".yaml")])
            densepose_models.extend([f for f in os.listdir(densepose_model_dir) if f.endswith((".pkl", ".pth"))])
        
        # If no models found, provide a way to input path directly or guide user
        if len(densepose_configs) == 1 : densepose_configs = ["path/to/your/config.yaml"] # Placeholder
        if len(densepose_models) == 1 : densepose_models = ["path/to/your/model.pkl"]   # Placeholder


        return {
            "required": {
                "image": ("IMAGE",),
                "category": (AVAILABLE_CATEGORIES, {"default": AVAILABLE_CATEGORIES[0] if AVAILABLE_CATEGORIES else "bottom"}),
                "prob_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "closing_kernel_size": ("INT", {"default": 0, "min": 0, "max": 51, "step": 1}), # Odd numbers preferred
                "dilation_kernel_size": ("INT", {"default": 0, "min": 0, "max": 51, "step": 1}),# Odd numbers preferred
                "apply_convex_hull": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                 "densepose_config_file": (densepose_configs,), # COMBO list
                 "densepose_model_file": (densepose_models,),   # COMBO list
                 "custom_dp_config_path": ("STRING", {"default": ""}), # Allow manual path input
                 "custom_dp_model_path": ("STRING", {"default": ""}),   # Allow manual path input
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute_parser"
    CATEGORY = "Image/Segment"

    def execute_parser(self, image: torch.Tensor, category: str,
                       prob_threshold: float, closing_kernel_size: int,
                       dilation_kernel_size: int, apply_convex_hull: bool,
                       densepose_config_file: str = None, densepose_model_file: str = None,
                       custom_dp_config_path: str = "", custom_dp_model_path: str = ""):

        # --- Initialize models if not already done ---
        if not initialize_segformer_pipeline():
            print("Segformer pipeline failed to initialize. Parsing mask will be empty.")
        
        dp_predictor = None
        actual_dp_config_path = custom_dp_config_path.strip()
        actual_dp_model_path = custom_dp_model_path.strip()

        if not actual_dp_config_path and densepose_config_file != "None" and densepose_config_file != "path/to/your/config.yaml":
            actual_dp_config_path = os.path.join(folder_paths.models_dir, "densepose", densepose_config_file)
        if not actual_dp_model_path and densepose_model_file != "None" and densepose_model_file != "path/to/your/model.pkl":
            actual_dp_model_path = os.path.join(folder_paths.models_dir, "densepose", densepose_model_file)

        use_densepose = _DENSEPOSE_AVAILABLE and os.path.exists(actual_dp_config_path) and os.path.exists(actual_dp_model_path)
        if use_densepose:
            dp_predictor = setup_densepose_predictor(actual_dp_config_path, actual_dp_model_path, prob_threshold)
            if dp_predictor is None:
                print("DensePose predictor failed to initialize. DensePose mask will be empty.")
                use_densepose = False # Fallback if predictor init fails
        else:
            if _DENSEPOSE_AVAILABLE and (category in CATEGORY_MAPPINGS and CATEGORY_MAPPINGS[category].get("densepose")):
                 print("DensePose config/model path not valid or not provided. DensePose part of masking will be skipped.")


        # --- Convert ComfyUI image tensor to PIL and OpenCV formats ---
        img_pil_rgb = tensor_to_pil(image)
        img_bgr_cv = cv2.cvtColor(np.array(img_pil_rgb), cv2.COLOR_RGB2BGR)
        H, W = img_bgr_cv.shape[:2]

        # --- Get category specific labels ---
        selected_map = CATEGORY_MAPPINGS.get(category, {})
        seg_labels = selected_map.get("segformer", [])
        dp_ids = selected_map.get("densepose", []) if use_densepose else []


        # --- Generate Segformer Parsing Mask ---
        parsing_mask_bool = np.zeros((H, W), dtype=bool)
        if seg_labels and (SEGFORMER_PIPELINE and SEGFORMER_PIPELINE != "failed_to_init"):
            print("Node: Generating Segformer parsing mask...")
            parsing_mask_bool = generate_parsing_mask_node(img_pil_rgb, seg_labels)
        else:
            print("Node: Skipping Segformer (no labels or pipeline issue).")

        # --- Generate Smoother DensePose Binary Mask ---
        smoother_dp_mask_bool = np.zeros((H, W), dtype=bool)
        if use_densepose and dp_ids and dp_predictor:
            print("Node: Running DensePose inference...")
            densepose_outputs = run_densepose_inference(img_bgr_cv, dp_predictor)
            if densepose_outputs:
                print("Node: Generating Smoother DensePose mask...")
                smoother_dp_mask_bool = generate_smoother_densepose_binary_mask_node(
                    (H, W), densepose_outputs, dp_ids, prob_threshold
                )
        else:
            print("Node: Skipping DensePose (not enabled, no IDs, or predictor issue).")


        # --- Combine Masks ---
        # Ensure shapes match before combining (though they should if generated from same H,W)
        if smoother_dp_mask_bool.shape != parsing_mask_bool.shape:
            print(f"Warning: Mask shapes mismatch! DP: {smoother_dp_mask_bool.shape}, Segformer: {parsing_mask_bool.shape}. Resizing Segformer mask.")
            parsing_mask_bool_resized = cv2.resize(parsing_mask_bool.astype(np.uint8),
                                               (smoother_dp_mask_bool.shape[1], smoother_dp_mask_bool.shape[0]),
                                               interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            parsing_mask_bool_resized = parsing_mask_bool
            
        processed_mask_bool = np.logical_or(smoother_dp_mask_bool, parsing_mask_bool_resized)
        processed_mask_uint8 = processed_mask_bool.astype(np.uint8) * 255

        # --- Apply Convex Hull ---
        if apply_convex_hull:
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
        if closing_kernel_size > 0:
            eff_close_k = closing_kernel_size + (1 - closing_kernel_size % 2) # Ensure odd
            print(f"Node: Applying closing with kernel {eff_close_k}x{eff_close_k}...")
            closing_kernel = np.ones((eff_close_k, eff_close_k), np.uint8)
            processed_mask_uint8 = cv2.morphologyEx(processed_mask_uint8, cv2.MORPH_CLOSE, closing_kernel)

        if dilation_kernel_size > 0:
            eff_dilate_k = dilation_kernel_size + (1 - dilation_kernel_size % 2) # Ensure odd
            print(f"Node: Applying dilation with kernel {eff_dilate_k}x{eff_dilate_k}...")
            dilation_kernel = np.ones((eff_dilate_k, eff_dilate_k), np.uint8)
            processed_mask_uint8 = cv2.dilate(processed_mask_uint8, dilation_kernel, iterations=1)
        
        # --- Convert final mask to ComfyUI MASK tensor ---
        final_mask_tensor = np_mask_to_tensor(processed_mask_uint8.astype(bool))

        return (final_mask_tensor,)


NODE_CLASS_MAPPINGS = {
    "ComfyPoseParser": ComfyPoseParserNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyPoseParser": "Pose Parser (DensePose+Segformer)"
}
