# Vision Deployment Pipeline: PyTorch → ONNX → TensorRT (FP32 and FP16)

A hands-on project taking a PyTorch vision model through the standard deployment path used in industrial inference (such as AOI systems): export to ONNX, build TensorRT engines in FP32 and FP16, then verify correctness, measure latency, and check accuracy on real images.

## Notebooks
- `01_pytorch_onnx_benchmark.ipynb`: first pass through the pipeline (PyTorch → ONNX → TensorRT FP32).
- `02_fair_benchmark.ipynb`: clean rerun with an interleaved benchmark in a single session, the FP16 engine, and a real-image accuracy check. **The results below come from this notebook.**

## Setup
- Hardware: Google Colab GPU (NVIDIA T4)
- Software: Python 3.13, PyTorch, ONNX Runtime (CUDA), TensorRT 11.5
- Model: ResNet18 (torchvision, pretrained, not fine-tuned)
- Input: 1x3x224x224, batch size 1
- Latency method: 20 warm-up runs, then 100 timed runs, repeated 5 times with all runtimes interleaved in one session (mean ± std)
- Accuracy data: 1000 random images (seeded) from the Imagenette validation split, resized to 256, center-cropped to 224, ImageNet-normalised

## Pipeline
1. **Export** the PyTorch model to ONNX (opset 17, dynamic batch axis).
2. **Verify** that ONNX Runtime outputs match PyTorch.
3. **Build** a TensorRT FP32 engine from the ONNX file with the TensorRT Python API.
4. **Verify** that the engine outputs match PyTorch.
5. **Export** a half-precision ONNX model and build a TensorRT FP16 engine.
6. **Check** FP16 output error and agreement against FP32.
7. **Benchmark** all runtimes.
8. **Measure accuracy** of PyTorch, TensorRT FP32 and TensorRT FP16 on real images.

## Results

### Latency

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
| FP16 top-1 agreement with FP32, random inputs | 50 / 50 |

### Accuracy on real images (1000 Imagenette validation images, 1000-way argmax)

| Runtime | Accuracy |
|---|---|
| PyTorch FP32 | 66.8% (668/1000) |
| TensorRT FP32 | 66.8% (668/1000) |
| TensorRT FP16 | 66.7% (667/1000) |

FP16 changed the predicted class on 3 of 1000 images relative to FP32.

### Engine size

| Engine | Size |
|---|---|
| TensorRT FP32 | 76.3 MB |
| TensorRT FP16 | 23.6 MB |

FP16 was about 2.6x faster than FP32 on the same engine and about 3.2x smaller on disk.

## Notes and caveats
- **Corrected benchmark:** my first run (notebook 01) used PyTorch without `torch.inference_mode()` and measured the runtimes at different times. It showed larger speedups (TensorRT FP32 at 2.78x), inflated by PyTorch's training-mode overhead. The results above fix both problems.
- **PyTorch baseline is noisy:** its latency ranged from about 2.9 to 3.5 ms across runs, so speedups relative to PyTorch are approximate. The TensorRT timings were much more stable. The FP16 vs FP32 ratio is the more reliable comparison.
- **Accuracy metric understates task accuracy:** the model predicts among all 1000 ImageNet classes while the data covers 10, so a related but different label counts as an error. The weakest class was church (25%): the model most often predicted "monastery" (34 of 92 images) and the correct "church" class came second (23). The same scoring was applied to all runtimes, so the FP32 vs FP16 comparison is unaffected.
- **FP16 accuracy scope:** the FP16 result (one image difference, within the statistical noise of a 1000-image test) applies to ResNet18 classification. A model that detects fine defects should be validated separately.
- **FP16 in TensorRT 11:** the `BuilderFlag.FP16` flag was removed and all networks are strongly typed, so I exported the PyTorch model in half precision and let the ONNX data types drive the engine precision.
- **Benchmark fairness:** the ONNX Runtime timing includes a CPU-to-GPU input copy on every call, while PyTorch and TensorRT keep data on the GPU. ONNX Runtime was run in FP32 only, so the FP32 vs FP16 comparison applies to TensorRT.
- **Shared GPU:** Colab GPUs are shared, so absolute latency varies between sessions. Compare runtimes within a session, not across sessions.
- **GPU-specific engines:** TensorRT engines are built for the GPU they were created on and are not portable to other GPU models.

## What I learned
- Exporting a model without errors does not guarantee identical outputs, so I check the output difference at every stage.
- Benchmarks need warm-up runs and GPU synchronisation, because GPU work is asynchronous.
- Fair benchmarks need the baseline set up properly: using `inference_mode` and running all runtimes in one session changed my first results noticeably.
- Reduced precision trades accuracy for speed and memory, so the error has to be measured on real data, and an absolute difference is only meaningful relative to the size of the outputs.
- A surprising metric deserves investigation: a per-class breakdown and a look at the model's actual predictions explained the low church accuracy.
- Library versions change APIs. TensorRT 11 removed APIs that most tutorials still use, so I read the migration guide and adapted the code.
- pip TensorRT does not include `trtexec` or the C++ headers, so I built the engines with the Python API.

## Next steps
- Run the FP16 ONNX model in ONNX Runtime for a like-for-like comparison
- Score accuracy over only the 10 relevant classes
- C++ inference with the TensorRT engine
- Apply the pipeline to a defect-detection model (MVTec AD)
- Package the pipeline in a Docker container
