from transformers.onnx import export
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Path to fine-tuned model
model_path = "./phi2_woocommerce_model"

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Export to ONNX
export(
    preprocessor=tokenizer,
    model=model,
    config=model.config,
    opset=13,
    output=Path("onnx_phi2_model")
)

print("✅ Model exported to ONNX format at ./onnx_phi2_model/")
