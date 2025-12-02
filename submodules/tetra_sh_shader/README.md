# tetra_sh_shader

Standalone CMake target 和 Python 包，用于体素化/八叉树的基础示例（无 OpenGL 依赖）。C++ 部分提供 VoxelNode/VoxelOctree 基类并通过 pybind11 暴露给 Python。

## Build (C++)
```bash
cd submodules/tetra_sh_shader
cmake -S . -B build
cmake --build build
```

### System dependencies
- CMake >= 3.16
- C++17 编译器（无额外库依赖）

## Python package
`pyproject.toml` 指向 `python/tetra_sh_shader/__init__.py`，通过 pybind11 绑定 C++ 体素类。
```bash
pip install -e submodules/tetra_sh_shader --no-build-isolation  # 需要已有 pybind11 头文件
python - <<'PY'
import tetra_sh_shader as t
tree = t.VoxelOctree()
print(tree)
PY
```
