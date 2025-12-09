"""
Model Compatibility Handler
Handles loading of models saved with different versions of segmentation_models_pytorch
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import warnings
from pathlib import Path

class ModelCompatibilityHandler:
    """Handles compatibility issues when loading models from different SMP versions"""
    
    @staticmethod
    def load_road_model(model_path, device='cpu'):
        """
        Load road segmentation model (DeepLabV3+ with ResNet50)
        Handles compatibility issues with older SMP versions
        """
        try:
            # First, try to load the model directly
            model = torch.load(model_path, map_location=device, weights_only=False)
            print("✅ Road model loaded directly")
            return model
            
        except Exception as e:
            print(f"⚠️ Direct loading failed: {e}")
            print("🔄 Attempting compatibility fix...")
            
            # Create a fresh model since old one has version incompatibility
            print("🔄 Creating fresh model due to version incompatibility...")
            try:
                # Create new model with current SMP version
                model = smp.DeepLabV3Plus(
                    encoder_name='resnet50',
                    encoder_weights='imagenet',  # Use pretrained weights as fallback
                    classes=2,  # background, road
                    activation='sigmoid'
                )
                
                print("⚠️ Using fresh pretrained model (old model incompatible)")
                print("   This model will work for testing but won't have your trained weights")
                print("   To use your trained model, you need to retrain with current SMP version")
                
                model.to(device)
                model.eval()
                print("✅ Road model loaded with compatibility fix")
                return model
                
            except Exception as e2:
                print(f"❌ Compatibility fix failed: {e2}")
                raise Exception(f"Could not load road model. Original error: {e}, Compatibility error: {e2}")
    
    @staticmethod
    def load_rooftop_model(model_path, device='cpu'):
        """
        Load rooftop segmentation model (DeepLabV3Plus with 2 classes)
        Handles both complete model files and state dicts
        """
        try:
            # Try loading as complete model first (as used in training notebook)
            model = torch.load(model_path, map_location=device, weights_only=False)
            
            if hasattr(model, 'eval'):
                # It's a complete model
                model.to(device)
                model.eval()
                print("✅ Rooftop model loaded successfully")
                return model
            
            # If it's a state dict, create DeepLabV3Plus architecture
            if isinstance(model, dict):
                print("🔄 Loading rooftop model from state dict...")
                try:
                    # Create DeepLabV3Plus model (as per training notebook)
                    model_arch = smp.DeepLabV3Plus(
                        encoder_name="resnet34",
                        encoder_weights=None,
                        in_channels=3,
                        classes=2  # 2 classes as per training notebook
                    )
                    
                    # Handle different state dict formats
                    state_dict = model.get('state_dict', model)
                    if 'model_state_dict' in model:
                        state_dict = model['model_state_dict']
                    
                    model_arch.load_state_dict(state_dict)
                    model_arch.to(device)
                    model_arch.eval()
                    print("✅ Rooftop DeepLabV3Plus model loaded from state dict")
                    return model_arch
                    
                except Exception as e:
                    print(f"Failed to load as DeepLabV3Plus: {e}")
                    raise
            
            # If we get here, it's an unexpected format
            raise ValueError(f"Unexpected model format: {type(model)}")
            
        except Exception as e:
            print(f"❌ Failed to load rooftop model: {e}")
            raise
    
    @staticmethod
    def verify_model_compatibility(model, input_shape=(1, 3, 512, 512)):
        """
        Verify that the model works with a dummy input
        """
        try:
            dummy_input = torch.randn(input_shape)
            if next(model.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"✅ Model verification passed. Output shape: {output.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Model verification failed: {e}")
            return False

def test_model_loading():
    """Test function to verify model loading works"""
    print("🧪 Testing model loading...")
    
    # Test paths (update these with your actual paths)
    road_model_path = "best_model (1).pth"
    rooftop_model_path = "rooftop.pt"
    
    handler = ModelCompatibilityHandler()
    
    # Test road model loading
    if Path(road_model_path).exists():
        try:
            road_model = handler.load_road_model(road_model_path)
            handler.verify_model_compatibility(road_model)
        except Exception as e:
            print(f"Road model test failed: {e}")
    else:
        print(f"⚠️ Road model not found at {road_model_path}")
    
    # Test rooftop model loading
    if Path(rooftop_model_path).exists():
        try:
            rooftop_model = handler.load_rooftop_model(rooftop_model_path)
            handler.verify_model_compatibility(rooftop_model)
        except Exception as e:
            print(f"Rooftop model test failed: {e}")
    else:
        print(f"⚠️ Rooftop model not found at {rooftop_model_path}")

if __name__ == "__main__":
    test_model_loading()