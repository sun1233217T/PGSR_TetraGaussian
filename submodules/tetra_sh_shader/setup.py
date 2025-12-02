from pathlib import Path

from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError as exc:
    raise RuntimeError("Building tetra_sh_shader_cpp now requires PyTorch to be installed.") from exc

root = Path(__file__).parent
readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""

sources = [
    "src/bindings.cpp",
    "src/voxel_node.cpp",
    "src/voxel_grid.cpp",
    "src/pre_resterization.cpp",
    "src/voxel_utility.cpp",
    "src/voxel_initialize.cpp",
    "src/rasterizer_torch.cpp",
]
cuda_sources = [
    "src/rasterizer_cuda.cu",
]

ext_modules = [
    CUDAExtension(
        "tetra_sh_shader_cpp",
        sources=sources + cuda_sources,
        extra_compile_args={
            "cxx": ["-fopenmp"],
            "nvcc": ["-lineinfo"],
        },
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="tetra-sh-shader",
    version="0.1.0",
    description="CMake sample and Python helpers for tetrahedral SH-style shading experiments.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="PGSR",
    author_email="*****@sdu.edu.com",
    url="sunnytom.cc",
    python_requires=">=3.8",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    include_package_data=True,
    install_requires=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)


# pip install -e submodules/tetra_sh_shader --no-build-isolation
