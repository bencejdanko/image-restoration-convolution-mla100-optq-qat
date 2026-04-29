# Image Restoration with Convolutional Models on the MLA100 using OPTQ and QAT

We train and evaluate image restoration models, preserving maximum fidelity of the original FP16 weights while quantizing to INT8 to run on the Mobilint MLA100 NPU. We benchmark for PSNR differences and latency deltas.

<img width="1780" height="612" alt="image" src="https://github.com/user-attachments/assets/6f734a3c-8680-4c08-b225-000f1076467e" />

## Final Results

| Model | Parameters | FP PSNR gain | INT8 PSNR Gain | CPU (Intel AI Boost) Latency | NPU (MLA100) Latency  | quantization_mode | quantization_method | quantization_output  | percentile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| ResNet | 10.71M | +1.10 | +1.07 | 1970.19ms | 418.43ms | maxPercentile (2) | WChAMulti (1) | Layer (0) | 0.999 | 
| Convolutional Network | 0.19M | +0.94 | +0.91 | 26.76ms | 5.49ms | maxPercentile (2) | WChAMulti (1) | Layer (0) | 0.999 | 
| Convolutional Network v2[^2] | 0.19M | +0.91 | +0.63 | 10.39ms | 5.68ms | maxPercentile (2) | WChAMulti (1) | Layer (0) | 0.999 |
| Convolutional Network v2[^1] | 0.19M | +0.91 | +0.66 | 10.39ms | 5.68ms | maxPercentile (2) | WChAMulti (1) | Layer (0) | 0.999 |
| Convolutional Network v3[^3] | 0.19M | +0.96 | +0.93 | 10.39ms | 5.68ms | maxPercentile (2) | WChAMulti (1) | Layer (0) | 0.999 |
| Convolutional Network v4[^4] | 0.37M | +1.05 | +1.03 | 24.37ms | 7.68ms | maxPercentile (2) | WChAMulti (1) | Layer (0) | 0.999 |
| Convolutional Network v5[^5] | 0.34M | +1.04 | +1.02 | 12.95ms | 6.62ms | maxPercentile (2) | WChAMulti (1) | Layer (0) | 0.999 |



1. Set with OPTQ and quantization-aware-training (4 epochs). We saw an increase in PSNR, though when applying OPTQ and MOD on good fidelity quantization, the score actually seems to stay the same/decrease.
2. Performance dropped significantly, but when swapped from LeakyReLU to regular ReLU, performance returned to be roughly equivalent. This was interesting, since the no-residual ConvNet did not have this issue.
3. We swapped LeakyReLU activations with ReLU, recovering PSNR fidelity when converting. However, our PSNR gains still plateau just under our requirements. Observationally, it seems that 0.19M params cannot physically store all the information needed for better restoration capacity.
4. We add residuals to our upscaling ops, and additional convolutional layers within each block after activations.
5. Removing refinement layer after upscaling

All gains from a baseline of 23.13 dB. An improvement of at least +1.05 +/- 0.05 is satisfactory for this dataset.

The training and validation set can be found on huggingface, at https://huggingface.co/datasets/bdanko/image-restoration-v2.

Models may be found at https://huggingface.co/bdanko/image-restoration and https://huggingface.co/bdanko/image-restoration-low-latency.

## Findings

- The NPU is extremely sensitive to feature declarations and spaces. Increasing feature spaces from 64 -> 65 will incure major latency costs because the NPU has to struggle to organize the kernel space.
- Some models, such as UNet-style bottlenecks with residuals, suffer a lot without the right quantization method.

## Quantization Modeling

Ultimately, our goal is to efficiently map a wide range of floating-point values ($FP16$) into discrete bucket of integers ($Int8$). The MLA100 NPU accelerates $Int8$ tensors operations and runs them extremely fast at low power. 

We use the proprietary `qubee` library from Mobilint to conduct quantization. There's an extensive configuration available for trying different quantization techniques:

### `quantization_mode`

This setting determines how the quantization range ($[min, max]$) is calculated from the activation distributions seen during calibration.

| Value | Name | Description |
| :--- | :--- | :--- |
| **0** | `percentile` | Uses the `percentile` parameter (default `0.9999`) to clip the top fraction of values. This is robust against extreme outliers. |
| **1** | `max` | Uses the absolute maximum value found in the calibration data. Simple, but highly sensitive to even a single outlier. |
| **2** | `maxPercentile` | **(Default)** A hybrid mode that often uses `topk_ratio` (default `0.01`). It typically calculates a "robust max" by looking at the top-K statistics. |
| **3** | `fastPercentile` | A computationally optimized approximation of the percentile method, used for faster calibration of very large models. |
| **4** | `histogramKL` | Minimizes the **Kullback-Leibler Divergence** (KL Divergence) between the original float distribution and the quantized distribution (similar to TensorRT's "Entropy" calibrator). |
| **5** | `histogramMSE` | Minimizes the **Mean Squared Error** (MSE) between the original and quantized values. |

### `quantization_method`

This determines whether scales are shared and if zero-points (asymmetric quantization) are used.

| Value | Name | Description |
| :--- | :--- | :--- |
| **0** | `WChALayer` | **W**eight per-**Ch**annel, **A**ctivation per-**Layer**. Every channel of weight has its own scale, but all activations in a layer share one scale. |
| **1** | `WChAMulti` | **(Default)** Similar to `WChALayer`, but activations share scales across **multiple layers**. This is critical for NPU optimization in residual blocks to avoid extra "Rescale" operations between additions. |
| **2** | `WChALayerZeropoint` | Same as `WChALayer` (0), but includes a **Zero-point** (Asymmetric). Essential for activations that aren't centered around zero (like ReLU or Sigmoid). |
| **3** | `WChAMultiZeropoint` | Same as `WChAMulti` (1), but with **Zero-points**. |

### `quantization_output`

This specifies how the output tensors of an operation are quantized.

| Value | Name | Description |
| :--- | :--- | :--- |
| **0** | `Layer` | **(Default)** Per-layer quantization for the output. |
| **1** | `Ch` | Per-channel quantization for the output. |
| **2** | `Sigmoid` | Specialized quantization range optimized for the $[0, 1]$ or $[-1, 1]$ output of Sigmoid/Tanh activations. |

### Advanced quantization technique options

#### `search_weight_scale`

The library conducts a branched search for best weight configurations. Parameters are hard coded in a C++ module, so it's difficult to ablate. It looks at a range of potential scaling factors (e.g., between 0 and 1), and measures how much the output of a specific layer changes (the "reconstruction error") when using different scales. It selects the scale that minimizes the difference between the original FP16 output and the new quantized output.

#### Quantization-aware-training with Minimum Output Difference (MOD): 

The model conducts additional training rounds, where you set epochs and a learning rate. The model trains to minimize outputs between inputs/outputs on the calibration data.

#### One-shot Post-Training Quantization (OPTQ)

The model iterates through weight blocks and adjust the remaining unquantized weights to compensate for the error introduced by the weights already quantized. The algorithm iterates and quantizes weights one block at a time, a size which you configure. You can set percentage damping (how sensitive the model is to weight changes).

#### Equivalent Transformations

There are options that can toggle mathematical smoothing of a model's weights before the actual quantization (like OPTQ) occurs. The objective is to scale layers and change the distribution of the weights, allowing them to be more quantization-friendly.

There are several options you can toggle:

```
| Rotation Transformations:                                                   |
|   [ ] SpinR1                                                                |
|   [ ] SpinR2                                                                |
|   [ ] QK Rotation                                                           |
|   [ ] Head OutCh Rotation

| Attention & FFN Transformations:                                            |
|   [ ] QK (Query-Key)                                                        |
|   [ ] VO (Value-Output)                                                     |
|   [ ] UD (Up-Down)                                                          |
|   [Y] NormConv                                                              | --> we can use for convolutional networks

| Quantization Enhancements:                                                  |
|   [ ] LUT Rescale                                                           |
|   [ ] FeedForward MultiLUT                                                  |
|   [ ] Flatten Quant                                                         |
|   [ ] Split FFN         
```
