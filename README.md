# Vision Deployment Pipeline: PyTorch → ONNX → TensorRT

A hands-on project taking a PyTorch vision model through the standard deployment path used in industrial inference (such as AOI systems): export to ONNX, build a TensorRT engine, then verify correctness and measure latency at each stage.

## Notebooks
- `01_pytorch_onnx_benchmark.ipynb`: first pass through the pipeline (PyTorch → ONNX → TensorRT).
- `02_fair_benchmark.ipynb`: clean rerun of the full pipeline with an interleaved benchmark in a single session. **The results below come from this notebook.**

## Setup
- Hardware: Google Colab GPU (NVIDIA T4)
- Software: Python 3.13, PyTorch, ONNX Runtime (CUDA), TensorRT 11.5
- Model: ResNet18 (torchvision, pretrained)
- Input: 1x3x224x224, batch size 1
- Method: 20 warm-up runs, then 100 timed runs, repeated 5 times with all three runtimes interleaved in one session (mean ± std)

## Pipeline
1. **Export** the PyTorch model to ONNX (opset 17, dynamic batch axis).
2. **Verify** that ONNX Runtime outputs match PyTorch.
3. **Build** a TensorRT engine from the ONNX file with the TensorRT Python API.
4. **Verify** that the engine outputs match PyTorch.
5. **Benchmark** all three runtimes.

## Results

| Runtime | Latency (ms) | Speedup vs PyTorch | Max abs diff vs PyTorch |
|---|---|---|---|
| PyTorch (GPU, inference_mode) | 2.92 ± 1.22 | 1.00x | n/a |
| ONNX Runtime (CUDA) | 2.20 ± 0.06 | 1.33x | 3.16e-6 |
| TensorRT (FP32) | 1.28 ± 0.01 | 2.29x | 4.29e-6 |

## Notes and caveats
- **Corrected benchmark:** my first run (notebook 01) used PyTorch without `torch.inference_mode()` and measured the runtimes at different times. It showed larger speedups (TensorRT 2.78x), which were inflated by PyTorch's training-mode overhead. The results above fix both problems.
- **PyTorch variance:** PyTorch's timing had a large spread (± 1.22 ms), so its speedup ratios are approximate.
- **Benchmark fairness:** the ONNX Runtime timing includes a CPU-to-GPU input copy on every call, while the PyTorch and TensorRT timings keep data on the GPU. Part of the gap between ONNX Runtime and TensorRT comes from this.
- **Shared GPU:** Colab GPUs are shared, so absolute latency varies between sessions. Compare runtimes within a session, not across sessions.
- **GPU-specific engine:** the TensorRT engine is built for the GPU it was created on and is not portable to other GPU models.
- **FP16 not yet tested:** TensorRT 11 removed the `BuilderFlag.FP16` build flag and makes all networks strongly typed, so reduced precision requires converting the ONNX model before building. This is planned next.

## What I learned
- Exporting a model without errors does not guarantee identical outputs, so I check the output difference at every stage.
- Benchmarks need warm-up runs and GPU synchronisation, because GPU work is asynchronous.
- Fair benchmarks need the baseline set up properly: using `inference_mode` and running all runtimes in one session changed my first results noticeably.
- Library versions change APIs. TensorRT 11 removed APIs that most tutorials still use, so I read the migration guide and adapted the code.
- pip TensorRT does not include `trtexec` or the C++ headers, so I built the engine with the Python API.

## Next steps
- FP16 engine and an accuracy-versus-latency comparison
- C++ inference with the TensorRT engine
- Apply the pipeline to a defect-detection model (MVTec AD)
- Package the pipeline in a Docker container
