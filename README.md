# Vision Deployment Pipeline: PyTorch → ONNX → TensorRT (FP32 and FP16)

A hands-on project taking a PyTorch vision model through the standard deployment path used in industrial inference (such as AOI systems): export to ONNX, build TensorRT engines in FP32 and FP16, then verify correctness and measure latency at each stage.

## Notebooks
- `01_pytorch_onnx_benchmark.ipynb`: first pass through the pipeline (PyTorch → ONNX → TensorRT FP32).
- `02_fair_benchmark.ipynb`: clean rerun with an interleaved benchmark in a single session, plus the FP16 engine. **The results below come from this notebook.**

## Setup
- Hardware: Google Colab GPU (NVIDIA T4)
- Software: Python 3.13, PyTorch, ONNX Runtime (CUDA), TensorRT 11.5
- Model: ResNet18 (torchvision, pretrained)
- Input: 1x3x224x224, batch size 1
- Method: 20 warm-up runs, then 100 timed runs, repeated 5 times with all runtimes interleaved in one session (mean ± std)

## Pipeline
1. **Export** the PyTorch model to ONNX (opset 17, dynamic batch axis).
2. **Verify** that ONNX Runtime outputs match PyTorch.
3. **Build** a TensorRT FP32 engine from the ONNX file with the TensorRT Python API.
4. **Verify** that the engine outputs match PyTorch.
5. **Export** a half-precision ONNX model and build a TensorRT FP16 engine.
6. **Check** FP16 output error and top-1 agreement against FP32.
7. **Benchmark** all runtimes.

## Results

| Runtime | Latency (ms) | Speedup vs PyTorch |
|---|---|---|
| PyTorch (GPU, inference_mode) | 3.51 ± 0.94 | 1.00x |
| ONNX Runtime (CUDA, FP32) | 2.28 ± 0.15 | 1.54x |
| TensorRT (FP32) | 1.29 ± 0.01 | 2.71x |
| TensorRT (FP16) | 0.50 ± 0.00 | 7.07x |

### Correctness

| Check | Result |
|---|---|
| ONNX Runtime vs PyTorch, max abs diff | 3.16e-6 |
| TensorRT FP32 vs PyTorch, max abs diff | 4.29e-6 |
| TensorRT FP16 vs FP32 PyTorch, max abs diff | 0.0196 (0.31% relative to the largest output) |
| FP16 top-1 agreement with FP32 | 50 / 50 random inputs |

### Engine size

| Engine | Size |
|---|---|
| TensorRT FP32 | 76.3 MB |
| TensorRT FP16 | 23.6 MB |

FP16 was about 2.6x faster than FP32 on the same engine and about 3.2x smaller on disk.

## Notes and caveats
- **Corrected benchmark:** my first run (notebook 01) used PyTorch without `torch.inference_mode()` and measured the runtimes at different times. It showed larger speedups (TensorRT FP32 at 2.78x), inflated by PyTorch's training-mode overhead. The results above fix both problems.
- **PyTorch baseline is noisy:** its latency ranged from about 2.9 to 3.5 ms across runs, so speedups relative to PyTorch are approximate. The TensorRT timings were much more stable (± 0.01 ms or less). The FP16 vs FP32 ratio is the more reliable comparison.
- **FP16 accuracy is not fully validated:** the correctness checks used random inputs, not real images. Before using FP16 in production, accuracy should be compared on real labelled data, especially for tasks that detect fine defects.
- **FP16 in TensorRT 11:** the `BuilderFlag.FP16` flag was removed and all networks are strongly typed, so I exported the PyTorch model in half precision and let the ONNX data types drive the engine precision.
- **Benchmark fairness:** the ONNX Runtime timing includes a CPU-to-GPU input copy on every call, while PyTorch and TensorRT keep data on the GPU. ONNX Runtime was also run in FP32 only, so the FP32 vs FP16 comparison applies to TensorRT.
- **Shared GPU:** Colab GPUs are shared, so absolute latency varies between sessions. Compare runtimes within a session, not across sessions.
- **GPU-specific engines:** TensorRT engines are built for the GPU they were created on and are not portable to other GPU models.

## What I learned
- Exporting a model without errors does not guarantee identical outputs, so I check the output difference at every stage.
- Benchmarks need warm-up runs and GPU synchronisation, because GPU work is asynchronous.
- Fair benchmarks need the baseline set up properly: using `inference_mode` and running all runtimes in one session changed my first results noticeably.
- Reduced precision trades accuracy for speed and memory, so the error has to be measured, and an absolute difference is only meaningful relative to the size of the outputs.
- Library versions change APIs. TensorRT 11 removed APIs that most tutorials still use, so I read the migration guide and adapted the code.
- pip TensorRT does not include `trtexec` or the C++ headers, so I built the engines with the Python API.

## Next steps
- Validate FP16 accuracy on real labelled images
- Run the FP16 ONNX model in ONNX Runtime for a like-for-like comparison
- C++ inference with the TensorRT engine
- Apply the pipeline to a defect-detection model (MVTec AD)
- Package the pipeline in a Docker container
