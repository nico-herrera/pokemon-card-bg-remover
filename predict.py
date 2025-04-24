import os
import torch
import numpy as np
from PIL import Image
from typing import List, Union
from cog import BasePredictor, Input, Path
from skimage import transform, morphology

# Import the fixed U2NET class
from u2net_model import U2NET

class Predictor(BasePredictor):
    def setup(self):
        """
        Load the U2-Net model
        """
        # Model loading happens here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Define model path
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, 'isnet-general-use.pth')
        
        # Check if model file exists
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Model file not found at {model_file}. "
                "Please make sure to include the model file in the models directory."
            )
        
        # Load model
        print("Loading model...")
        self.model = U2NET()
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")

    def predict(
        self,
        image: Path = Input(description="Pokemon card image to remove background from"),
        color_threshold: float = Input(
            description="Threshold for identifying colored regions (0.0-1.0)", 
            default=0.5
        ),
        preserve_colors: bool = Input(
            description="Specially preserve colored areas of the card",
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
        Remove background from Pokemon card image, with special focus on preserving colored card elements
        """
        # Load and prepare image
        print(f"Processing image: {image}")
        input_image = Image.open(image).convert("RGB")
        
        # Preprocess image
        img_tensor = self.preprocess(input_image)
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(img_tensor)
        
        # Postprocess
        mask = self.postprocess(prediction, input_image.size[::-1])
        
        # Special handling for colored regions if enabled
        if preserve_colors:
            mask = self.enhance_color_preservation(input_image, mask, color_threshold)
        
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
    
    def enhance_color_preservation(self, image, mask, threshold=0.5):
        """
        Enhance the mask to better preserve colored regions of Pokemon cards
        
        This is especially useful for keeping the colorful borders and 
        artwork of Pokemon cards while removing the background
        """
        # Convert to numpy arrays
        img_np = np.array(image)
        mask_np = np.array(mask)
        
        # Create a color saturation map
        hsv = self.rgb_to_hsv(img_np)
        saturation = hsv[:, :, 1]
        
        # Find regions with high color saturation (likely card artwork and borders)
        color_regions = saturation > threshold
        
        # Dilate color regions to ensure complete coverage
        color_regions = morphology.binary_dilation(color_regions, morphology.disk(3))
        
        # Combine with original mask for better color preservation
        combined_mask = np.maximum(mask_np, color_regions * 255).astype(np.uint8)
        
        # Clean up the combined mask
        combined_mask = morphology.opening(combined_mask, morphology.disk(2))
        
        return Image.fromarray(combined_mask)
    
    def rgb_to_hsv(self, rgb):
        """Convert RGB image to HSV color space"""
        # Normalize RGB to 0-1
        rgb_norm = rgb.astype(np.float32) / 255.0
        
        # Get RGB channels
        r, g, b = rgb_norm[:,:,0], rgb_norm[:,:,1], rgb_norm[:,:,2]
        
        # Calculate Value (max)
        v = np.maximum(np.maximum(r, g), b)
        
        # Calculate delta
        delta = v - np.minimum(np.minimum(r, g), b)
        
        # Initialize saturation
        s = np.zeros_like(v)
        # Avoid division by zero
        s[v != 0] = delta[v != 0] / v[v != 0]
        
        # Initialize hue
        h = np.zeros_like(v)
        
        # Calculate hue
        mask = delta != 0
        h[mask & (v == r)] = 60 * ((g[mask & (v == r)] - b[mask & (v == r)]) / delta[mask & (v == r)] % 6)
        h[mask & (v == g)] = 60 * ((b[mask & (v == g)] - r[mask & (v == g)]) / delta[mask & (v == g)] + 2)
        h[mask & (v == b)] = 60 * ((r[mask & (v == b)] - g[mask & (v == b)]) / delta[mask & (v == b)] + 4)
        
        # Normalize hue to 0-1
        h = h / 360.0
        
        # Stack channels
        hsv = np.stack([h, s, v], axis=2)
        
        return hsv
    
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