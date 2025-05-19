# 🧠 Phi-2 WooCommerce Product Trainer + ONNX Export

This project fine-tunes Microsoft's `phi-2` language model on WooCommerce product data and exports the model to ONNX for fast deployment.

## 🔧 Steps

1. Convert WooCommerce product data to training format (`train.txt`)
2. Fine-tune Phi-2 (`train_phi2.py`)
3. Export to ONNX (`convert_to_onnx.py`)
4. Run fast inference with ONNX Runtime

## 🧪 Inference Sample

```python
import onnxruntime as ort
# Coming soon...
