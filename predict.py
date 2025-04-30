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
    ) -> Path:
        """
        Remove background from Pokemon card image, preserving the card itself
        """
        # Load image
        print(f"Processing image: {image}")
        input_image = Image.open(image).convert("RGB")
        
        # Use contour-based method to detect card boundaries
        if use_contour_method:
            mask = self.detect_card_boundary(input_image)
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
    
    def detect_card_boundary(self, image):
        """
        Improved method to detect Pokemon card boundaries in an image
        
        Args:
            image: PIL Image of the Pokemon card with background
            
        Returns:
            mask: Binary mask where the card is white (255) and background is black (0)
        """
        import cv2
        
        # Convert PIL image to OpenCV format
        img_cv = np.array(image)
        img_cv = img_cv[:, :, :3]  # Remove alpha channel if present
        
        # Get image dimensions
        height, width = img_cv.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Try multiple approaches and choose the best result
        masks = []
        
        # Approach 1: Adaptive thresholding to separate card from background
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 51, 10)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask from the largest contour
        mask1 = np.zeros((height, width), dtype=np.uint8)
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour to reduce noise
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # If it's approximately a rectangle (4 points) or has reasonable area
            contour_area = cv2.contourArea(largest_contour)
            if len(approx) <= 8 and contour_area > (height * width * 0.2):
                # Draw filled contour on mask
                cv2.drawContours(mask1, [approx], 0, 255, -1)
        
        masks.append(mask1)
        
        # Approach 2: Otsu thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours in the Otsu thresholded image
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask from the largest contour
        mask2 = np.zeros((height, width), dtype=np.uint8)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            contour_area = cv2.contourArea(largest_contour)
            if contour_area > (height * width * 0.2):
                cv2.drawContours(mask2, [approx], 0, 255, -1)
        
        masks.append(mask2)
        
        # Approach 3: Simple rectangular detection
        # Blur image to reduce noise and apply edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect them
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours in the dilated edges
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask from the rectangular approximation of the largest contour
        mask3 = np.zeros((height, width), dtype=np.uint8)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Only accept if the rectangle is large enough (at least 30% of the image)
            if w * h > (height * width * 0.3):
                cv2.rectangle(mask3, (x, y), (x + w, y + h), 255, -1)
        
        masks.append(mask3)
        
        # Approach 4: Try to find the most rectangular contour
        mask4 = np.zeros((height, width), dtype=np.uint8)
        if contours and len(contours) > 0:
            # Calculate how rectangular each contour is
            best_contour = None
            best_score = 0
            min_area = height * width * 0.2  # Minimum area requirement
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                
                # Perfect rectangle would have ratio of 1
                if rect_area > 0:
                    rectangularity = area / rect_area
                    # Pokemon cards are rectangular with rounded corners
                    # so rectangularity should be high but not 1
                    if 0.7 < rectangularity < 1.0 and rectangularity > best_score:
                        best_score = rectangularity
                        best_contour = contour
            
            if best_contour is not None:
                # Get the rectangular bounds
                x, y, w, h = cv2.boundingRect(best_contour)
                cv2.rectangle(mask4, (x, y), (x + w, y + h), 255, -1)
        
        masks.append(mask4)
        
        # Choose the best mask based on coverage and shape
        best_mask = None
        best_score = 0
        
        for mask in masks:
            # Skip empty masks
            if np.max(mask) == 0:
                continue
                
            # Calculate mask coverage
            coverage = np.sum(mask) / (255 * height * width)
            
            # Only consider masks that cover a reasonable portion of the image
            if 0.3 <= coverage <= 0.98:
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Calculate rectangularity
                    area = cv2.contourArea(largest_contour)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    rect_area = w * h
                    
                    if rect_area > 0:
                        rectangularity = area / rect_area
                        # Score based on combination of coverage and rectangularity
                        score = rectangularity * (coverage if coverage <= 0.9 else (1.8 - coverage))
                        
                        if score > best_score:
                            best_score = score
                            best_mask = mask
        
        # If we found a good mask, use it
        if best_mask is not None:
            return Image.fromarray(best_mask)
        
        # Fallback: If all approaches failed, return a simple threshold-based mask
        _, simple_thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        return Image.fromarray(simple_thresh)
    
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
    
    def apply_mask(self, image, mask):
        """Apply mask to original image for transparent background"""
        # Ensure image is RGBA
        image = image.convert("RGBA")
        
        # Convert mask to numpy array if it's a PIL Image
        if isinstance(mask, Image.Image):
            mask_np = np.array(mask)
        else:
            mask_np = mask
            
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
            alpha_smooth = morphology.opening(alpha, morphology.disk(int(sigma)))
            img_array[:, :, 3] = alpha_smooth
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(img_array)
        
        return enhanced_image