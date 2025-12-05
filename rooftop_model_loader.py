"""
Rooftop Model Loader
Handles loading and inference for the rooftop segmentation model
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
import os

# Import from current directory
from model_compatibility import ModelCompatibilityHandler
from rooftop_preprocessing import RooftopPreprocessor

class RooftopModelLoader:
    """Handles rooftop model loading and inference"""
    
    def __init__(self, model_path="rooftop.pt", device='cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.preprocessor = RooftopPreprocessor()
        
        # Class mapping for rooftop segmentation (2 classes from training notebook)
        self.class_names = [
            'background',    # 0
            'rooftop'        # 1 - Rooftop class
        ]
        
        # Define which classes are considered "rooftops"
        self.rooftop_classes = [1]  # Rooftop class
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the rooftop segmentation model"""
        print(f"🏠 Loading rooftop model from {self.model_path}")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Rooftop model not found at {self.model_path}")
        
        # Use compatibility handler to load model
        handler = ModelCompatibilityHandler()
        self.model = handler.load_rooftop_model(self.model_path, self.device)
        
        # Verify model works
        if not handler.verify_model_compatibility(self.model):
            raise RuntimeError("Rooftop model verification failed")
        
        print("✅ Rooftop model loaded and verified successfully")
    
    def predict(self, image_path_or_array, threshold=0.05):
        """
        Predict rooftop mask for input image with patch-based inference for large images
        
        Args:
            image_path_or_array: Path to image or numpy array
            threshold: Threshold for rooftop probability
        
        Returns:
            dict: Contains 'mask', 'class_mask', 'raw_output', 'original_image'
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Load original image
        if isinstance(image_path_or_array, str):
            try:
                original_image = np.array(Image.open(image_path_or_array).convert("RGB"))
            except:
                original_image = cv2.cvtColor(cv2.imread(image_path_or_array), cv2.COLOR_BGR2RGB)
        else:
            original_image = image_path_or_array.copy()
        
        # Check if image is large and needs patch-based inference
        height, width = original_image.shape[:2]
        patch_size = 256
        
        if height > patch_size * 1.5 or width > patch_size * 1.5:
            # Use patch-based inference for large images
            print(f"  🔍 Large image detected ({height}x{width}), using patch-based inference...")
            rooftop_mask = self._predict_patches(original_image, threshold, patch_size)
        else:
            # Use standard inference for small images
            rooftop_mask = self._predict_single(original_image, threshold)
        
        # Create dummy outputs for compatibility
        class_predictions = rooftop_mask.copy()
        prob_np = rooftop_mask.astype(np.float32)
        
        return {
            'mask': rooftop_mask,
            'class_mask': class_predictions,
            'raw_output': prob_np,
            'original_image': original_image,
            'tensor_image': None
        }
    
    def _predict_single(self, original_image, threshold):
        """Standard inference for small images"""
        # Preprocess image
        tensor_image, _ = self.preprocessor.preprocess_image(original_image)
        tensor_image = tensor_image.to(self.device)
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            raw_output = self.model(tensor_image)
        
        # Process output
        probabilities = torch.softmax(raw_output, dim=1)
        prob_np = probabilities.detach().cpu().numpy()
        
        if len(prob_np.shape) == 4:
            prob_np = prob_np[0]
        
        if prob_np.shape[0] == 2:
            rooftop_prob = prob_np[1]
            rooftop_mask = (rooftop_prob > threshold).astype(np.uint8)
        else:
            class_predictions = np.argmax(prob_np, axis=0)
            rooftop_mask = np.zeros_like(class_predictions, dtype=np.uint8)
            for rooftop_class in self.rooftop_classes:
                rooftop_mask[class_predictions == rooftop_class] = 1
        
        # Resize back to original size
        if rooftop_mask.shape != original_image.shape[:2]:
            rooftop_mask = cv2.resize(rooftop_mask, (original_image.shape[1], original_image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        
        return rooftop_mask
    
    def _predict_patches(self, original_image, threshold, patch_size):
        """Patch-based inference for large images"""
        height, width = original_image.shape[:2]
        
        # Calculate number of patches needed
        patches_h = (height + patch_size - 1) // patch_size
        patches_w = (width + patch_size - 1) // patch_size
        
        # Initialize output mask
        full_mask = np.zeros((height, width), dtype=np.uint8)
        
        print(f"    Processing {patches_h}x{patches_w} = {patches_h * patches_w} patches...")
        
        for i in range(patches_h):
            for j in range(patches_w):
                # Calculate patch boundaries
                y1 = i * patch_size
                y2 = min((i + 1) * patch_size, height)
                x1 = j * patch_size
                x2 = min((j + 1) * patch_size, width)
                
                # Extract patch
                patch = original_image[y1:y2, x1:x2]
                
                # Pad patch to patch_size if needed
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded_patch
                
                # Run inference on patch
                patch_mask = self._predict_single(patch, threshold)
                
                # Place result back in full mask (crop if padded)
                actual_h = y2 - y1
                actual_w = x2 - x1
                full_mask[y1:y2, x1:x2] = patch_mask[:actual_h, :actual_w]
        
        return full_mask
    
    def predict_batch(self, image_list, threshold=0.5):
        """
        Predict rooftop masks for batch of images
        
        Args:
            image_list: List of image paths or numpy arrays
            threshold: Threshold for probability-based classes
        
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        for image in image_list:
            result = self.predict(image, threshold)
            results.append(result)
        return results
    
    def visualize_prediction(self, prediction_result, overlay_alpha=0.5, show_all_classes=False):
        """
        Create visualization of rooftop prediction
        
        Args:
            prediction_result: Result from predict() method
            overlay_alpha: Alpha for mask overlay
            show_all_classes: If True, show all classes; if False, only rooftops
        
        Returns:
            np.ndarray: Visualization image
        """
        original = prediction_result['original_image']
        
        if show_all_classes:
            # Show all classes with different colors
            class_mask = prediction_result['class_mask']
            
            # Resize mask to match original image size
            if class_mask.shape != original.shape[:2]:
                class_mask = cv2.resize(class_mask, (original.shape[1], original.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # Create colored mask for all classes
            colored_mask = np.zeros_like(original)
            colors = [
                [0, 0, 0],       # background - black
                [255, 0, 0]      # rooftop - red
            ]
            
            for class_id, color in enumerate(colors):
                if class_id < len(colors):
                    colored_mask[class_mask == class_id] = color
        
        else:
            # Show only rooftops
            rooftop_mask = prediction_result['mask']
            
            # Resize mask to match original image size
            if rooftop_mask.shape != original.shape[:2]:
                rooftop_mask = cv2.resize(rooftop_mask, (original.shape[1], original.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            
            # Create colored mask (red for rooftops)
            colored_mask = np.zeros_like(original)
            colored_mask[rooftop_mask > 0] = [255, 0, 0]  # Red for rooftops
        
        # Overlay on original image
        visualization = cv2.addWeighted(original, 1-overlay_alpha, colored_mask, overlay_alpha, 0)
        
        return visualization
    
    def get_rooftop_statistics(self, prediction_result):
        """
        Get statistics about rooftop detection
        
        Args:
            prediction_result: Result from predict() method
        
        Returns:
            dict: Statistics about rooftop detection
        """
        mask = prediction_result['mask']
        
        total_pixels = mask.size
        rooftop_pixels = np.sum(mask)
        rooftop_percentage = (rooftop_pixels / total_pixels) * 100
        
        return {
            'total_pixels': total_pixels,
            'rooftop_pixels': rooftop_pixels,
            'rooftop_percentage': rooftop_percentage,
            'image_shape': mask.shape
        }

def test_rooftop_model():
    """Test rooftop model loading and inference"""
    print("🧪 Testing rooftop model...")
    
    model_path = "rooftop.pt"
    
    if not Path(model_path).exists():
        print(f"⚠️ Rooftop model not found at {model_path}")
        print("Please ensure the model file exists to run this test")
        return False
    
    try:
        # Initialize model loader
        loader = RooftopModelLoader(model_path)
        
        # Test with TIFF image if available
        tiff_image_path = "image dhairya.tiff"
        if Path(tiff_image_path).exists():
            print(f"📸 Testing with TIFF image: {tiff_image_path}")
            test_image = tiff_image_path
        else:
            print("📸 TIFF image not found, using dummy image")
            # Create dummy image for testing
            test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Test prediction
        result = loader.predict(test_image)
        
        print("✅ Rooftop model test successful")
        print(f"   Rooftop mask shape: {result['mask'].shape}")
        print(f"   Rooftop mask unique values: {np.unique(result['mask'])}")
        print(f"   Class mask shape: {result['class_mask'].shape}")
        print(f"   Class mask unique values: {np.unique(result['class_mask'])}")
        
        # Test statistics
        stats = loader.get_rooftop_statistics(result)
        print(f"   Rooftop pixels: {stats['rooftop_pixels']}")
        print(f"   Rooftop percentage: {stats['rooftop_percentage']:.2f}%")
        
        # Test visualization and save it
        viz = loader.visualize_prediction(result)
        print(f"   Visualization shape: {viz.shape}")
        
        # Save visualization if using real image
        if isinstance(test_image, str):
            import cv2
            output_path = "rooftop_detection_result.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
            print(f"   💾 Saved visualization: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rooftop model test failed: {e}")
        return False

if __name__ == "__main__":
    test_rooftop_model()