# Tested environment

The supplied source archive was executed in an environment named `driver` with:

- Python 3.10
- NVIDIA RTX 3060, 6 GB
- cuDNN 9
- PyTorch 2.11.0 + CUDA 12.6 build
- torchvision 0.26.0 + CUDA 12.6 build
- Ultralytics 8.4.87
- NumPy 1.26.4
- pandas 2.3.3
- SciPy 1.15.3
- Matplotlib 3.10.8
- OpenCV-Python 4.9.0.80

The original `pip freeze` contained many packages unrelated to this project, so the public repository uses a minimal dependency list instead of publishing the full environment snapshot.
