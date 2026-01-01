import torch
import torch.nn as nn
from torchvision import models

def get_model():
    # Re-define structure to load weights
    model = models.mobilenet_v3_small()
    original_layer = model.features[0][0]
    model.features[0][0] = nn.Conv2d(1, original_layer.out_channels, 
                          kernel_size=original_layer.kernel_size, 
                          stride=original_layer.stride, 
                          padding=original_layer.padding, bias=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    return model

def export():
    # 1. Load the trained PyTorch model
    model = get_model()
    try:
        model.load_state_dict(torch.load("robocon_model.pth"))
        model.eval() # Set to evaluation mode
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: robocon_model.pth not found. Run train.py first.")
        return

    # 2. Create dummy input (Batch=1, Channel=1, Height=224, Width=224)
    dummy_input = torch.randn(1, 1, 224, 224)

    # 3. Export to ONNX
    output_file = "robocon_vision.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        output_file, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    print(f"SUCCESS: Model exported to '{output_file}'")

if __name__ == "__main__":
    export()