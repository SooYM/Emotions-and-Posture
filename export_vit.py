import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

def export_model():
    model_id = "mo-thecreator/vit-Facial-Expression-Recognition"
    output_dir = "public/vit_onnx"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading processor and model for {model_id}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    
    print("Saving configurations...")
    processor.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    
    print("Exporting model to ONNX...")
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=14,
        do_constant_folding=True
    )
    print(f"Successfully exported ONNX model to {onnx_path}!")

if __name__ == "__main__":
    export_model()
