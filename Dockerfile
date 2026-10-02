# Genesis and Quadrants on ROCm 10.0.0.
#
# Build from the root of this Genesis repository. The build context is this
# tree. Quadrants is cloned during the build; it is not part of the context.
#
#   docker build -t genesis-release-rocm10 .
#
# Override the Quadrants checkout with:
#
#   docker build \
#     --build-arg QUADRANTS_REPO=https://github.com/AMD-Ecosystem/quadrants.git \
#     --build-arg QUADRANTS_REF=rocm10.0.0_r26.10 \
#     -t genesis-release-rocm10 .
#
# Run on a host whose kernel driver matches ROCm 10, with the GPU devices passed in:
#
#   docker run --rm -it \
#     --device=/dev/kfd --device=/dev/dri \
#     --group-add "$(stat -c '%g' /dev/kfd)" \
#     genesis-release-rocm10
#
# The base image supplies the ROCm 10 userspace (/opt/rocm). Quadrants is compiled
# in the build stage with C++ tests on. The final image keeps the virtualenv,
# quadrants_cpp_tests, the Quadrants Python tests, and this Genesis tree.

ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:10.0.0-full@sha256:a90cf047f615abe70fbef83c64def0a2d549ef37a39c8ea545430aba4981b374

FROM ${ROCM_IMAGE} AS build

ARG QUADRANTS_REPO=https://github.com/AMD-Ecosystem/quadrants.git
ARG QUADRANTS_REF=rocm10.0.0_r26.10

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        build-essential \
        clang \
        cmake \
        ninja-build \
        git \
        ca-certificates \
        xz-utils \
        lsb-release \
        liblz4-dev \
        libssl-dev \
        libncurses-dev \
        libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv \
    && pip install -U "pip>=25.1"

# Clone the Quadrants branch that carries the spirv_codegen test link, then
# initialize its submodules. ./build.py downloads LLVM 22 and the Vulkan SDK,
# then builds the wheel. AMDGPU is off by default; CUDA is on by default and
# is turned off because this image has no CUDA toolkit. Vulkan stays off for
# a ROCm-only runtime. QD_BUILD_TESTS builds quadrants_cpp_tests.
RUN git clone --branch "${QUADRANTS_REF}" --single-branch "${QUADRANTS_REPO}" /src/quadrants

WORKDIR /src/quadrants
RUN git submodule update --init --recursive \
    && pip install --group dev \
    && CMAKE_ARGS="-DQD_WITH_AMDGPU:BOOL=ON -DQD_WITH_CUDA:BOOL=OFF -DQD_WITH_VULKAN:BOOL=OFF -DQD_BUILD_TESTS:BOOL=ON" \
       ./build.py wheel \
    && mkdir -p /opt/quadrants \
    && find /src/quadrants/build -name quadrants_cpp_tests -type f -exec cp {} /opt/quadrants/quadrants_cpp_tests \; \
    && test -x /opt/quadrants/quadrants_cpp_tests

# genesis-world pins quadrants==1.3.3. Install every other dependency from
# pyproject.toml first, so a later Genesis source edit does not rebuild Quadrants or
# reinstall PyTorch. Genesis itself is installed without dependencies so pip
# leaves the source-built Quadrants wheel in place.
COPY pyproject.toml /tmp/genesis-pyproject.toml

RUN python -c 'import tomllib; from pathlib import Path; \
data = tomllib.loads(Path("/tmp/genesis-pyproject.toml").read_text()); \
deps = data["project"]["dependencies"] + data["project"]["optional-dependencies"]["dev"]; \
kept = [dep for dep in deps if dep.split("[", 1)[0].split(";", 1)[0].strip().lower().replace("_", "-") != "quadrants"]; \
Path("/tmp/genesis-reqs.txt").write_text("\n".join(kept) + "\n")'

RUN pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
        "torch[device-all]==2.13.0+rocm10.0.0" \
    && pip install /src/quadrants/dist/quadrants-*.whl \
    && pip install -r /tmp/genesis-reqs.txt \
    && pip install --group test

WORKDIR /src/Genesis
COPY . /src/Genesis

# 32-lane scan. inclusive_add would cover the whole 64-lane CDNA wavefront.
# This branch already uses inclusive_add_tiled; this still rewrites a plain
# inclusive_add if one is present, then refuses to build without the 32-lane call.
RUN sed -i 's/qd.simt.subgroup.inclusive_add(value)/qd.simt.subgroup.inclusive_add_tiled(value, 5)/' \
        genesis/utils/simt.py \
    && grep -q 'inclusive_add_tiled(value, 5)' genesis/utils/simt.py \
    && pip install --no-deps -e /src/Genesis


FROM ${ROCM_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib:/opt/rocm/lib \
    ROCM_PATH=/opt/rocm \
    PYOPENGL_PLATFORM=egl \
    PYTHONPATH=/opt/debug-runtime:/opt/rocm/share/amd_smi \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        libpython3.12 \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libx11-6 \
        libxrender1 \
        libxext6 \
        libxi6 \
        libxrandr2 \
        libxcursor1 \
        ca-certificates \
        git \
        gcc \
        python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
COPY --from=build /src/Genesis /src/Genesis
COPY --from=build /opt/quadrants/quadrants_cpp_tests /opt/quadrants/quadrants_cpp_tests
COPY --from=build /src/quadrants/tests /src/quadrants/tests
COPY docker/sitecustomize.py /opt/debug-runtime/sitecustomize.py

# Quadrants dlopens "libamdhip64.so". Point that name at the ROCm runtime
# shipped inside the PyTorch wheel so the process does not load a second HIP.
RUN ln -sf libamdhip64.so.7 \
    /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so

WORKDIR /src/Genesis

CMD ["python", "-c", "import torch, quadrants, genesis; print('torch', torch.__version__); print('quadrants', quadrants.__version_str__); print('genesis', genesis.__file__)"]
