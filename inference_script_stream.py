import os
import time
import statistics
import numpy as np
from PIL import Image
import onnxruntime as ort
import maccel
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
MODEL_REPO = "bdanko/image-restoration-low-latency"
DATA_REPO = "bdanko/image-restoration-v2"

# --- Helper Functions ---
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def preprocess_image(pil_img):
    img = pil_img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
    
    onnx_input = arr.transpose(2, 0, 1) # HWC -> CHW
    onnx_input = np.expand_dims(onnx_input, axis=0) # CHW -> NCHW
    
    npu_input = arr # HWC
    return onnx_input, npu_input, arr

def postprocess_output(out_img):
    return np.clip(out_img, 0, 1)

def get_test_iterator(dataset_split):
    """Generator to process images one-by-one to save RAM."""
    for item in dataset_split:
        lr_onnx, lr_npu, hr_ground_truth = preprocess_image(item['lr'])
        # Reference is resized ground truth
        hr_ref = np.array(item['hr'].convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
        
        yield {
            'lr_onnx': lr_onnx,
            'lr_npu': lr_npu,
            'hr_ref': hr_ref,
            'baseline_psnr': calculate_psnr(hr_ground_truth, hr_ref)
        }

# --- 1. Downloads & Setup ---
print(f"Downloading models from {MODEL_REPO}...")
MXQ_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="convnet-v5.mxq")
ONNX_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="convnet-v5.onnx")

print(f"Loading validation split from {DATA_REPO}...")
# split="validation" ensures only that slice is cached
dataset = load_dataset(DATA_REPO, split="validation")

# --- 2. Initialize Inference Engines ---
print("Initializing CPU (ONNX Runtime)...")
cpu_session = ort.InferenceSession(ONNX_PATH)
cpu_input_name = cpu_session.get_inputs()[0].name

npu_ready = False
try:
    print("Initializing NPU (maccel)...")
    npu = maccel.Accelerator(0)
    npu_model = maccel.Model(MXQ_PATH)
    npu_model.launch(npu)
    npu_ready = True
except Exception as e:
    print(f"NPU not available: {e}")

# --- 3. Warmup ---
print("Warming up engines...")
first_item = next(get_test_iterator(dataset))
for _ in range(3):
    cpu_session.run(None, {cpu_input_name: first_item['lr_onnx']})
    if npu_ready:
        npu_model.infer([first_item['lr_npu']])

# --- 4. Main Combined Inference Loop ---
cpu_times, cpu_psnrs = [], []
npu_times, npu_psnrs = [], []
baseline_psnrs = []

print(f"Starting combined inference on {len(dataset)} images...")
# Total used for progress bar because we know it from len(dataset)
for sample in tqdm(get_test_iterator(dataset), total=len(dataset), desc="Processing"):
    baseline_psnrs.append(sample['baseline_psnr'])

    # CPU Run
    t0 = time.perf_counter()
    cpu_out = cpu_session.run(None, {cpu_input_name: sample['lr_onnx']})
    cpu_times.append((time.perf_counter() - t0) * 1000)
    
    cpu_img = postprocess_output(cpu_out[0][0].transpose(1, 2, 0))
    cpu_psnrs.append(calculate_psnr(cpu_img, sample['hr_ref']))

    # NPU Run
    if npu_ready:
        t1 = time.perf_counter()
        npu_out = npu_model.infer([sample['lr_npu']])[0]
        npu_times.append((time.perf_counter() - t1) * 1000)
        
        npu_img = postprocess_output(np.asarray(npu_out).astype(np.float32))
        npu_psnrs.append(calculate_psnr(npu_img, sample['hr_ref']))

# --- 5. Reporting ---
avg_baseline = statistics.mean(baseline_psnrs)
avg_cpu_time = statistics.mean(cpu_times)
avg_cpu_psnr = statistics.mean(cpu_psnrs)

print(f"\n{'='*30}\nFinal Results\n{'='*30}")
print(f"Images Processed: {len(baseline_psnrs)}")
print(f"Baseline (Input) PSNR: {avg_baseline:.2f} dB")
print(f"---")
print(f"CPU Avg Time: {avg_cpu_time:.2f} ms ({1000/avg_cpu_time:.1f} FPS)")
print(f"CPU Avg PSNR: {avg_cpu_psnr:.2f} dB (Gain: {avg_cpu_psnr - avg_baseline:+.2f})")

if npu_ready:
    avg_npu_time = statistics.mean(npu_times)
    avg_npu_psnr = statistics.mean(npu_psnrs)
    print(f"---")
    print(f"NPU Avg Time: {avg_npu_time:.2f} ms ({1000/avg_npu_time:.1f} FPS)")
    print(f"NPU Avg PSNR: {avg_npu_psnr:.2f} dB (Gain: {avg_npu_psnr - avg_baseline:+.2f})")
    print(f"---")
    print(f"Speedup: {avg_cpu_time/avg_npu_time:.2f}x faster on NPU")
    npu_model.dispose()
else:
    print("\nNPU results not available.")
