# pgsr_oct environment notes

Set up the `pgsr_oct` conda env with CUDA-aware PyTorch, OpenGL bindings, and common geometry math libs.

## Python packages
```bash
# Base environment
conda create -n pgsr_oct python=3.10 -y
conda activate pgsr_oct

# Install PyTorch matching your CUDA; example for CUDA 12.1
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Project extras (shared with this repo)
pip install -r requirements_pgsr_oct.txt

# Optional: editable install of the C++ test harness helpers
pip install -e submodules/tetra_sh_shader
```

## System dependencies for C++/OpenGL work
- CMake >= 3.16
- OpenGL development headers (`libgl1-mesa-dev` on Debian/Ubuntu)
- GLFW (`libglfw3-dev`)
- Eigen3 (`libeigen3-dev`)
- GLM (`libglm-dev`) or use the vendored copy at `submodules/diff-plane-rasterization/third_party/glm`

The new CMake sample lives in `submodules/tetra_sh_shader` and will pick up the vendored GLM automatically when installed.
