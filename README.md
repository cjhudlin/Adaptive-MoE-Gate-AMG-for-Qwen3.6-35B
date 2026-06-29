# Adaptive-MoE-Gate-AMG-for-Qwen3.6-35B
The Adaptive MoE Gate (AMG) is an inference-time modification to llama.cpp that introduces cumulative probability thresholding on expert routing weights. Rather than always using exactly k experts, AMG uses as many experts as needed to reach a confidence threshold
