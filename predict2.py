import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Union
from cog import BasePredictor, Input, Path
from skimage import transform, morphology

# You can still keep the U2NET model as a backup option
from u2net_model import U2NET

class Predictor(BasePredictor):
    def setup(self):
        """
        Load the model (optional if using contour-based method)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # For contour-based approach, we don't need to load a model
        # But we'll keep it as a backup option
        try:
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            model_file = os.path.join(model_dir, 'isnet-general-use.pth')
            
            if os.path.exists(model_file):
                print("Loading model as backup option...")
                self.model = U2NET()
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.model_available = True
                print("Model loaded successfully")
            else:
                self.model_available = False
                print("Model file not found, will use contour-based approach only")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_available = False

    def predict(
        self,
        image: Path = Input(description="Pokemon card image to remove background from"),
        use_contour_method: bool = Input(
            description="Use contour-based method for card detection", 
            default=True
        ),
        edge_smooth: int = Input(
            description="Amount of edge smoothing (0-5)",
            default=2,
            ge=0,
            le=5
        ),
        color_threshold: int = Input(
            description="Threshold for color-based background removal (0-255)",
            default=230,
            ge=0,
            le=255
        ),
        detect_method: str = Input(
            description="Card detection method to use",
            default="auto",
            choices=["auto", "threshold", "rectangle", "convex_hull", "grab_cut"]
        ),
    ) -> Path:
        """
        Remove background from Pokemon card image, preserving the card itself
        """
        # Load image
        print(f"Processing image: {image}")
        input_image = Image.open(image).convert("RGB")
        
        # Use correct detection for card boundaries
        if use_contour_method:
            mask = self.detect_card_boundary_multi(input_image, color_threshold, detect_method)
        elif self.model_available:
            # Fallback to model-based approach if requested
            img_tensor = self.preprocess(input_image)
            with torch.no_grad():
                prediction = self.model(img_tensor)
            mask = self.postprocess(prediction, input_image.size[::-1])
        else:
            # If model not available and contour method not selected
            raise ValueError("Model-based approach selected but model not available. Please use contour_method=True")
        
        # Apply mask to original image
        result = self.apply_mask(input_image, mask)
        
        # Enhance edges if requested
        if edge_smooth > 0:
            result = self.enhance_edges(result, edge_smooth)
        
        # Save result
        output_path = Path(os.path.join(os.getcwd(), "output.png"))
        result.save(output_path)
        print(f"Saved result to {output_path}")
        
        return output_path
    
    def detect_card_boundary_multi(self, image, color_threshold=230, method="auto"):
        """
        Multi-method approach to detect Pokemon card boundaries
        
        Args:
            image: PIL Image of the Pokemon card with background
            color_threshold: Threshold value for color-based background removal
            method: Detection method to use (auto, threshold, rectangle, convex_hull, grab_cut)
            
        Returns:
            mask: Binary mask where the card is white (255) and background is black (0)
        """
        # Convert PIL image to OpenCV format
        img_cv = np.array(image)
        img_cv = img_cv[:, :, :3]  # Remove alpha channel if present
        
        # Get image dimensions
        height, width = img_cv.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Dictionary to store masks from different methods
        masks = {}
        
        # Method 1: Simple thresholding
        if method == "auto" or method == "threshold":
            # Use Otsu's thresholding for adaptive threshold value
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # If specified threshold is provided, also try that
            if color_threshold != 0:
                _, thresh_custom = cv2.threshold(gray, color_threshold, 255, cv2.THRESH_BINARY_INV)
                
                # Choose the better of the two thresholds
                # (based on which one has a more reasonable card area)
                area_otsu = np.sum(thresh) / 255
                area_custom = np.sum(thresh_custom) / 255
                ideal_area = height * width * 0.5  # Assume card takes about half the image
                
                if abs(area_custom - ideal_area) < abs(area_otsu - ideal_area):
                    thresh = thresh_custom
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            masks["threshold"] = thresh
        
        # Method 2: Rectangular detection
        if method == "auto" or method == "rectangle":
            rect_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Try to find contours in the threshold mask
            if "threshold" in masks:
                contours, _ = cv2.findContours(masks["threshold"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the largest contour by area
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Get the rotated rectangle that bounds the contour
                    rect = cv2.minAreaRect(largest_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # Draw the rotated rectangle
                    cv2.drawContours(rect_mask, [box], 0, 255, -1)
            
            masks["rectangle"] = rect_mask
        
        # Method 3: Convex hull approach
        if method == "auto" or method == "convex_hull":
            # Apply edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 150)
            
            # Dilate edges to connect them
            kernel = np.ones((5, 5), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours in the dilated edges
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create convex hull mask
            hull_mask = np.zeros((height, width), dtype=np.uint8)
            
            if contours:
                # Combine all significant contours
                all_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > height * width * 0.01:  # Filter out tiny contours
                        all_contours.append(contour)
                
                if all_contours:
                    # Combine all contours into one
                    combined_contour = np.vstack(all_contours)
                    
                    # Get the convex hull of the combined contours
                    hull = cv2.convexHull(combined_contour)
                    
                    # Draw the hull on the mask
                    cv2.drawContours(hull_mask, [hull], 0, 255, -1)
            
            masks["convex_hull"] = hull_mask
        
        # Method 4: GrabCut approach
        if method == "auto" or method == "grab_cut":
            # Need a starting mask - use rectangle if available
            if "rectangle" in masks and np.sum(masks["rectangle"]) > 0:
                # Create a GrabCut mask
                gc_mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
                gc_mask[:] = cv2.GC_BGD  # Set everything to background initially
                
                # Mark the rectangle area as probable foreground
                rect_mask = masks["rectangle"]
                gc_mask[rect_mask > 0] = cv2.GC_PR_FGD
                
                try:
                    # Create temporary arrays for GrabCut
                    bgdModel = np.zeros((1, 65), np.float64)
                    fgdModel = np.zeros((1, 65), np.float64)
                    
                    # Define a rectangle containing the probable foreground
                    contours, _ = cv2.findContours(rect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        rect = (x, y, w, h)
                        
                        # Apply GrabCut
                        cv2.grabCut(img_cv, gc_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                        
                        # Create the final mask
                        grabcut_mask = np.zeros((height, width), dtype=np.uint8)
                        grabcut_mask[np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))] = 255
                        
                        masks["grab_cut"] = grabcut_mask
                except Exception as e:
                    print(f"GrabCut error: {e}")
                    # If GrabCut fails, don't use this method
                    pass
        
        # Choose the best mask based on the specified method or auto-detection
        final_mask = None
        
        if method != "auto":
            # Use the specified method if available
            if method in masks:
                final_mask = masks[method]
        else:
            # Auto-select the best mask based on quality metrics
            best_score = 0
            
            for method_name, mask in masks.items():
                # Skip empty masks
                if np.sum(mask) == 0:
                    continue
                
                # Calculate some quality metrics
                # 1. Coverage (percentage of image covered by mask)
                coverage = np.sum(mask) / (255 * height * width)
                
                # 2. Compactness (ratio of area to perimeter squared, normalized)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                else:
                    compactness = 0
                
                # Score based on reasonable coverage (not too small, not too large)
                # and compactness (closer to 1 is better for card-like shapes)
                coverage_score = 1.0 - abs(coverage - 0.5) * 2  # Penalize deviation from 50% coverage
                
                # Pokemon cards are rectangular with rounded corners, so compactness should be high
                compactness_score = compactness
                
                # Calculate final score with weights
                score = (coverage_score * 0.4) + (compactness_score * 0.6)
                
                # Apply method-specific bonuses
                if method_name == "rectangle":
                    score *= 1.1  # Slight bonus for rectangle method
                elif method_name == "grab_cut":
                    score *= 1.2  # Higher bonus for GrabCut (most sophisticated)
                
                if score > best_score:
                    best_score = score
                    final_mask = mask
        
        # Fallback if no good mask was found or selected
        if final_mask is None or np.sum(final_mask) == 0:
            # Simple threshold fallback
            _, final_mask = cv2.threshold(gray, color_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Apply morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        return Image.fromarray(final_mask)
    
    def apply_mask(self, image, mask):
        """Apply mask to original image for transparent background"""
        # Ensure image is RGBA
        image = image.convert("RGBA")
        
        # Convert mask to numpy array if it's a PIL Image
        if isinstance(mask, Image.Image):
            mask_np = np.array(mask)
        else:
            mask_np = mask
            
        # Apply slight Gaussian blur for smoother edges
        mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0.8)
            
        # Create alpha channel from mask
        alpha = Image.fromarray(mask_np)
        
        # Apply alpha channel to image
        image.putalpha(alpha)
        
        return image
    
    def enhance_edges(self, image, edge_smooth=2):
        """
        Enhance the edges of the segmented card for a cleaner look
        
        Args:
            image: PIL Image with transparent background
            edge_smooth: Amount of edge smoothing (higher = smoother)
            
        Returns:
            PIL Image with enhanced edges
        """
        # Convert to numpy array with alpha channel
        img_array = np.array(image)
        
        # Get alpha channel
        alpha = img_array[:, :, 3]
        
        # Apply Gaussian blur to the alpha channel for smoother edges
        if edge_smooth > 0:
            # Scale edge_smooth parameter to an appropriate sigma value
            sigma = edge_smooth * 0.5
            
            # Create a temporary copy of alpha for smoothing
            alpha_float = alpha.astype(np.float32)
            
            # Apply Gaussian blur
            alpha_smooth = cv2.GaussianBlur(alpha_float, (0, 0), sigma)
            
            # Adjust semi-transparent pixels at the edge
            # This makes edge transitions smoother while keeping the card interior fully opaque
            edge_mask = (alpha > 10) & (alpha < 245)
            if np.any(edge_mask):
                alpha[edge_mask] = alpha_smooth[edge_mask]
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(img_array)
        
        return enhanced_image
    
    def preprocess(self, image):
        """Preprocess image for the model"""
        # Resize maintaining aspect ratio
        w, h = image.size
        ratio = min(1024/w, 1024/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        
        # Resize image
        img = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Convert to tensor and normalize
        img_np = np.array(img) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0)
        
        # Normalize with ImageNet means and stds
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.to(self.device)
    
    def postprocess(self, pred, original_size):
        """Process model output to create alpha mask"""
        # Convert to numpy
        pred = pred.squeeze().cpu().data.numpy()
        
        # Resize to original image size
        mask = transform.resize(pred, original_size, mode='constant')
        
        # Threshold to improve mask quality
        mask = (mask > 0.5).astype(np.float32)
        
        # Clean up mask with morphological operations
        mask = morphology.remove_small_holes(mask.astype(bool), 50)
        mask = morphology.remove_small_objects(mask.astype(bool), 50)
        mask = mask.astype(np.float32)
        
        # Normalize mask to 0-255
        mask = (mask * 255).astype(np.uint8)
        
        return Image.fromarray(mask)