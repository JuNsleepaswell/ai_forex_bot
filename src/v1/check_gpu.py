# check_gpu.py
import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
print("\n--- GPU Detection ---")

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        details = tf.config.experimental.get_device_details(gpu)
        print(f"    Compute Capability: {details.get('compute_capability')}")
        print(f"    Device Name: {details.get('device_name')}")
else:
    print("FATAL ERROR: No GPU was detected by TensorFlow.")
    print("Please ensure you have the correct NVIDIA drivers and CUDA-compatible TensorFlow installed.")

print("\nVerification complete.")