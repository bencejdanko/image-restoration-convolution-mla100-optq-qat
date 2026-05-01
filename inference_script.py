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

MODEL_REPO = "bdanko/image-restoration-low-latency"
DATA_REPO = "bdanko/image-restoration-v2"

print(f"Downloading models from {MODEL_REPO}...")
MXQ_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="convnet-v5.mxq")
ONNX_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="convnet-v5.onnx")

print(f"Loading validation set from {DATA_REPO}...")
dataset = load_dataset(DATA_REPO, split="validation")

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
    
    # HWC format for NPU
    npu_input = arr # HWC
    
    return onnx_input, npu_input, arr

def postprocess_output(out_img):
    return np.clip(out_img, 0, 1)

# Prepare test data
test_samples = []

print(f"Preparing {len(dataset)} samples from the full validation set...")
for i in range(len(dataset)):
    item = dataset[i]
    lr_onnx, lr_npu, hr_ground_truth = preprocess_image(item['lr'])
    hr_ref = np.array(item['hr'].convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
    test_samples.append({
        'lr_onnx': lr_onnx,
        'lr_npu': lr_npu,
        'hr_ref': hr_ref,
        'baseline_psnr': calculate_psnr(hr_ground_truth, hr_ref)
    })

baseline_mean_psnr = statistics.mean([s['baseline_psnr'] for s in test_samples])

print("Starting CPU Inference...")
cpu_session = ort.InferenceSession(ONNX_PATH)
input_name = cpu_session.get_inputs()[0].name

# Warmup
for _ in range(3):
    cpu_session.run(None, {input_name: test_samples[0]['lr_onnx']})

cpu_times = []
cpu_psnrs = []

for sample in tqdm(test_samples, desc="CPU"):
    start = time.perf_counter()
    outputs = cpu_session.run(None, {input_name: sample['lr_onnx']})
    elapsed = (time.perf_counter() - start) * 1000
    cpu_times.append(elapsed)
    
    # Output is NCHW, convert back to HWC
    out_img = outputs[0][0].transpose(1, 2, 0)
    out_img = postprocess_output(out_img)
    cpu_psnrs.append(calculate_psnr(out_img, sample['hr_ref']))

cpu_mean_time = statistics.mean(cpu_times)
cpu_mean_psnr = statistics.mean(cpu_psnrs)

print("Starting NPU Inference...")
try:
    npu = maccel.Accelerator(0)
    model = maccel.Model(MXQ_PATH)
    model.launch(npu)

    # Warmup
    for _ in range(3):
        model.infer([test_samples[0]['lr_npu']])

    npu_times = []
    npu_psnrs = []

    for sample in tqdm(test_samples, desc="NPU"):
        start = time.perf_counter()
        out = model.infer([sample['lr_npu']])[0]
        elapsed = (time.perf_counter() - start) * 1000
        npu_times.append(elapsed)
        
        out_img = np.asarray(out).astype(np.float32)
        out_img = postprocess_output(out_img)
        npu_psnrs.append(calculate_psnr(out_img, sample['hr_ref']))

    npu_mean_time = statistics.mean(npu_times)
    npu_mean_psnr = statistics.mean(npu_psnrs)
    model.dispose()

    results_ready = True
except Exception as e:
    print(f"NPU Inference failed (probably no NPU hardware present on this machine): {e}")
    results_ready = False

print(f"\nFinal Results on {len(test_samples)} images:")
print(f"Baseline input PSNR (no model): {baseline_mean_psnr:.2f} dB")
print(f"CPU Performance: {cpu_mean_time:.2f} ms/image ({1000/cpu_mean_time:.1f} img/sec), Mean PSNR: {cpu_mean_psnr:.2f} dB (Improvement: {cpu_mean_psnr - baseline_mean_psnr:+.2f} dB)")
if results_ready:
    print(f"NPU Performance: {npu_mean_time:.2f} ms/image ({1000/npu_mean_time:.1f} img/sec), Mean PSNR: {npu_mean_psnr:.2f} dB (Improvement: {npu_mean_psnr - baseline_mean_psnr:+.2f} dB)")
    print(f"NPU Speedup over CPU: {cpu_mean_time/npu_mean_time:.1f}x")
else:
    print("NPU results not available.")
