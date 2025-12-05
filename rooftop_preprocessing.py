"""
Rooftop Model Preprocessing
Based on the rooftop segmentation preprocessing pipeline
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class RooftopPreprocessor:
    """Preprocessing pipeline for rooftop segmentation model"""
    
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        
        # Define preprocessing parameters (from rooftop notebook)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Create preprocessing transform
        self.transform = self._get_preprocessing_transform()
    
    def _get_preprocessing_transform(self):
        """Create preprocessing transform pipeline"""
        return A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(transpose_mask=True)
        ])
    
    def preprocess_image(self, image_path_or_array):
        """
        Preprocess image for rooftop model inference
        
        Args:
            image_path_or_array: Path to image file or numpy array
        
        Returns:
            torch.Tensor: Preprocessed image tensor
            np.ndarray: Original image for visualization
        """
        # Load image if path is provided
        if isinstance(image_path_or_array, str):
            # Try different loading methods
            try:
                image = np.array(Image.open(image_path_or_array).convert("RGB"))
            except:
                image = cv2.cvtColor(cv2.imread(image_path_or_array), cv2.COLOR_BGR2RGB)
        else:
            image = image_path_or_array.copy()
        
        # Store original for visualization
        original_image = image.copy()
        
        # Apply preprocessing
        transformed = self.transform(image=image)
        processed_image = transformed['image']
        
        # Add batch dimension
        tensor_image = processed_image.unsqueeze(0)
        
        return tensor_image, original_image
    
    def preprocess_batch(self, image_list):
        """
        Preprocess a batch of images
        
        Args:
            image_list: List of image paths or numpy arrays
        
        Returns:
            torch.Tensor: Batch of preprocessed images
            list: List of original images
        """
        processed_images = []
        original_images = []
        
        for image in image_list:
            tensor_img, orig_img = self.preprocess_image(image)
            processed_images.append(tensor_img.squeeze(0))  # Remove batch dim
            original_images.append(orig_img)
        
        # Stack into batch
        import torch
        batch_tensor = torch.stack(processed_images)
        
        return batch_tensor, original_images
    
    def denormalize_image(self, tensor_image):
        """
        Denormalize image tensor back to [0, 255] range for visualization
        
        Args:
            tensor_image: Normalized tensor image
        
        Returns:
            np.ndarray: Denormalized image
        """
        # Convert tensor to numpy
        if tensor_image.dim() == 4:
            image = tensor_image.squeeze(0)
        else:
            image = tensor_image
        
        # Move to CPU if on GPU
        if image.is_cuda:
            image = image.cpu()
        
        # Convert to numpy and transpose
        image = image.numpy().transpose(1, 2, 0)
        
        # Denormalize
        mean = np.array(self.mean)
        std = np.array(self.std)
        image = (image * std + mean) * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image

# Helper function for compatibility with existing rooftop code
def get_rooftop_transforms(target_size=(256, 256)):
    """Get rooftop preprocessing transforms (compatible with original notebook)"""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    return A.Compose([
        A.Resize(target_size[0], target_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True)
    ])

def test_rooftop_preprocessing():
    """Test rooftop preprocessing pipeline"""
    print("🧪 Testing rooftop preprocessing...")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    
    # Initialize preprocessor
    preprocessor = RooftopPreprocessor()
    
    # Test preprocessing
    try:
        tensor_img, orig_img = preprocessor.preprocess_image(dummy_image)
        print(f"✅ Rooftop preprocessing successful")
        print(f"   Input shape: {dummy_image.shape}")
        print(f"   Output tensor shape: {tensor_img.shape}")
        print(f"   Original image shape: {orig_img.shape}")
        
        # Test denormalization
        denorm_img = preprocessor.denormalize_image(tensor_img)
        print(f"   Denormalized shape: {denorm_img.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Rooftop preprocessing failed: {e}")
        return False

if __name__ == "__main__":
    test_rooftop_preprocessing()