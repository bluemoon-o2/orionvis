import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from packaging.version import parse, Version
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

this_dir = Path(__file__).resolve().parent

FORCE_CXX11_ABI = os.getenv("FORCE_CXX11_ABI", "FALSE").upper() == "TRUE"
BUILD_CUDA_EXT = os.getenv("BUILD_CUDA_EXT", "AUTO").upper()
CUDA_ARCH_LIST = os.getenv("CUDA_ARCH_LIST", None)


def safe_check_output(cmd: List[str], env: Optional[Dict] = None) -> str:
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        warnings.warn(
            f"Command {' '.join(cmd)} failed with error: {e.stderr.strip()}",
            UserWarning
        )
        return ""
    except FileNotFoundError:
        warnings.warn(
            f"Command {' '.join(cmd)} failed: executable not found",
            UserWarning
        )
        return ""
    except subprocess.TimeoutExpired:
        warnings.warn(
            f"Command {' '.join(cmd)} timed out after 30 seconds",
            UserWarning
        )
        return ""


def get_cuda_bare_metal_version(cuda_dir: str) -> Tuple[Optional[str], Optional[Version]]:
    nvcc_path = Path(cuda_dir) / "bin" / "nvcc"
    if not nvcc_path.exists():
        nvcc_path = Path(cuda_dir) / "bin" / "nvcc.exe"
        if not nvcc_path.exists():
            warnings.warn(f"nvcc not found at {cuda_dir}/bin", UserWarning)
            return None, None

    raw_output = safe_check_output([str(nvcc_path), "-V"])
    if not raw_output:
        return None, None

    output = raw_output.split()
    try:
        release_idx = output.index("release") + 1
        version_str = output[release_idx].split(",")[0]
        bare_metal_version = parse(version_str)
        return raw_output, bare_metal_version
    except (ValueError, IndexError):
        warnings.warn(f"Failed to parse CUDA version from output: {raw_output}", UserWarning)
        return raw_output, None


def get_cuda_arch_flags() -> List[str]:
    cc_flags = []

    if CUDA_ARCH_LIST:
        arch_list = [arch.strip() for arch in CUDA_ARCH_LIST.split(",") if arch.strip()]
        for arch in arch_list:
            if arch.startswith("sm_"):
                compute_arch = arch.replace("sm_", "compute_")
                cc_flags.extend([f"-gencode", f"arch={compute_arch},code={arch}"])
            elif arch.startswith("compute_"):
                cc_flags.extend([f"-gencode", f"arch={arch},code={arch.replace('compute_', 'sm_')}"])
        return cc_flags

    cuda_home = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH") or CUDA_HOME
    if not cuda_home:
        return []

    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            print(f"Detected local GPU architecture: sm_{major}{minor}", flush=True)
            return [f"-gencode", f"arch=compute_{major}{minor},code=sm_{major}{minor}"]
        except Exception as e:
            print(f"Failed to detect local GPU architecture: {e}", flush=True)

    _, cuda_version = get_cuda_bare_metal_version(cuda_home)
    if not cuda_version:
        return []

    print(f"Detected CUDA version: {cuda_version}", flush=True)

    base_archs = []

    if cuda_version < Version("13.0"):
        base_archs.append(("7.0", "sm_70"))  # Volta

    base_archs.append(("8.0", "sm_80"))  # Ampere

    if cuda_version >= Version("11.8"):
        base_archs.append(("9.0", "sm_90"))  # Hopper
    if cuda_version >= Version("12.0"):
        base_archs.append(("9.2", "sm_92"))  # Ada Lovelace
        base_archs.append(("10.0", "sm_100"))  # Blackwell

    for compute_ver, sm_ver in base_archs:
        cc_flags.extend([f"-gencode", f"arch=compute_{compute_ver},code={sm_ver}"])

    return cc_flags


def get_compile_args() -> Dict[str, List[str]]:
    """获取编译参数，优化跨平台兼容性"""
    compile_args = {
        "cxx": [],
        "nvcc": []
    }

    # 通用C++编译参数
    if sys.platform == "win32":
        cxx_flags = [
            "/O2",
            "/std:c++17",
            "/W3",  # 适度警告级别
            "/EHsc",  # 异常处理
            "/DNOMINMAX",  # 避免Windows的min/max宏冲突
        ]
    else:
        cxx_flags = [
            "-O3",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-Wno-deprecated-declarations",
            "-fvisibility=hidden",
        ]
        # 添加针对GCC的优化
        if sys.platform.startswith("linux"):
            cxx_flags.extend([
                "-ffast-math",
                "-fno-finite-math-only",
                "-fopenmp",  # 启用OpenMP
            ])

    compile_args["cxx"] = cxx_flags

    # NVCC编译参数
    nvcc_flags = [
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=-v",
        "-lineinfo",
    ]

    # 平台特定NVCC参数
    if sys.platform == "win32":
        nvcc_flags.extend([
            "-Xcompiler", "/std:c++17",
            "-Xcompiler", "/EHsc",
            "-Xcompiler", "/DNOMINMAX",
        ])
    else:
        nvcc_flags.append("-std=c++17")
        if sys.platform.startswith("linux"):
            nvcc_flags.extend([
                "-Xcompiler", "-fopenmp",
                "-Xcompiler", "-fvisibility=hidden",
            ])

    # 添加CUDA架构标志
    nvcc_flags.extend(get_cuda_arch_flags())

    compile_args["nvcc"] = nvcc_flags

    return compile_args


def find_sources(root_dir, with_cuda=True):
    extensions = [".cpp", ".cu"] if with_cuda else [".cpp"]
    sources = []
    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext in extensions:
                sources.append(os.path.join(subdir, filename))
    return sources

def get_inplace_abn_ext() -> Optional[CUDAExtension]:
    if torch.has_cuda or os.getenv("IABN_FORCE_CUDA") == "1":
        compile_args = get_compile_args()
        return CUDAExtension(
            name="orionvis.extentions.inplace_abn._backend",
            sources=find_sources("orionvis/extentions/inplace_abn/src"),
            extra_compile_args=compile_args,
            include_dirs=[str(this_dir / "orionvis" / "extentions" / "inplace_abn" / "include")],
            define_macros=[("WITH_CUDA", 1)],
        )
    else:
        # Return a CppExtension or None if you want to support CPU-only builds
        return None

def get_ext_modules() -> List[CUDAExtension]:
    if BUILD_CUDA_EXT == "NO":
        warnings.warn(
            "BUILD_CUDA_EXT=NO is set. CUDA extensions will not be built.",
            UserWarning
        )
        return []

    cuda_available = torch.cuda.is_available()
    cuda_home = CUDA_HOME

    if BUILD_CUDA_EXT == "YES":
        if not cuda_available or not cuda_home:
            raise RuntimeError(
                "BUILD_CUDA_EXT=YES is set but CUDA is not available. "
                "Please install CUDA or set BUILD_CUDA_EXT=NO."
            )
    else:  # AUTO mode
        if not cuda_available or not cuda_home:
            warnings.warn(
                "CUDA is not available or CUDA_HOME is not set. "
                "The CUDA extension will not be built. "
                "The package will fall back to the pure PyTorch implementation.",
                UserWarning
            )
            return []

    # Debug print
    print(f"\n=== Build Configuration ===")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA_HOME: {cuda_home}")
    print(f"FORCE_CXX11_ABI: {FORCE_CXX11_ABI}")
    print(f"CUDA_ARCH_LIST: {CUDA_ARCH_LIST or 'auto-detected'}")
    print(f"===========================\n", flush=True)

    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
        print("Forced C++11 ABI compatibility (FORCE_CXX11_ABI=TRUE)", flush=True)

    compile_args = get_compile_args()

    ext_modules = []

    # Add inplace_abn extension
    inplace_abn_ext = get_inplace_abn_ext()
    if inplace_abn_ext:
        ext_modules.append(inplace_abn_ext)
    modes = ["oflex"]

    sources_map = {
        "oflex": [
            "orionvis/extentions/selective_scan/cusoflex/selective_scan_oflex.cpp",
            "orionvis/extentions/selective_scan/cusoflex/selective_scan_core_fwd.cu",
            "orionvis/extentions/selective_scan/cusoflex/selective_scan_core_bwd.cu",
        ],
    }

    include_dirs = [
        str(this_dir / "orionvis" / "extentions" / "selective_scan"),
    ]

    for mode in modes:
        ext_name = f"orionvis.extentions.selective_scan.selective_scan_cuda_{mode}"
        sources = sources_map[mode]
        print(f"Building selective_scan extension: {ext_name}")
        print(f"Sources: {sources}")

        missing_sources = [src for src in sources if not Path(src).exists()]
        if missing_sources:
            warnings.warn(
                f"Skipping {mode} extension: missing source files {missing_sources}",
                UserWarning
            )
            continue

        ext = CUDAExtension(
            name=ext_name,
            sources=sources,
            extra_compile_args=compile_args,
            include_dirs=include_dirs,
            libraries=["cudart"] if sys.platform != "win32" else [],
        )
        ext_modules.append(ext)

    return ext_modules


def read_readme():
    readme_path = this_dir / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "A Python package for computer vision tasks with integrated CUDA extensions."


setup(
    name="orionvis",
    packages=find_packages(include=["orionvis", "orionvis.*"]),
    package_dir={"": "."},
    package_data={
        "orionvis": [
            "configs/snaps/mobilemamba/*.txt",
            "configs/weights/*.yaml",
            "extentions/inplace_abn/include/*.h",
            "extentions/inplace_abn/include/*.cuh",
            "extentions/selective_scan/*.h",
            "extentions/selective_scan/*.cuh",
            "extentions/selective_scan/*.cpp",
            "extentions/selective_scan/*.cu",
        ]
    },
    author="bluemoon-o2",
    author_email="2095774200@shu.edu.cn",
    description="A Python package for computer vision tasks with integrated CUDA extensions.",
    long_description=read_readme(),
    long_description_content_type="markdown",
    url="https://github.com/bluemoon-o2/orionvis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Computer Vision",
        "Intended Audience :: Developers",
        "Intended Audience :: Researchers",
    ],
    keywords=["computer vision", "cuda", "pytorch", "deep learning", "selective scan"],
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3, <4",
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "write_to": "orionvis/_version.py",
    },
    install_requires=[
        "setuptools_scm",
        "torch>=2.0.0",
        "torchvision>=0.20.0",
        "packaging>=21.0",
        "ninja>=1.10.0",
        "einops>=0.6.0",
        "PyYAML>=6.0",
        "lazy_object_proxy>=1.12.0",
        "gdown>=5.0.0",
        "tqdm>=4.0.0",
        "timm>=0.9.0",
        "pywavelets>=1.1.1",
        "fvcore>=0.1.5",
    ],
    include_package_data=True,
    zip_safe=False,
)
