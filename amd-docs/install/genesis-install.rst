.. meta::
  :description: installing Genesis for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, Genesis

.. _genesis-on-rocm-installation:

********************************************************************
Genesis on ROCm installation
********************************************************************

System requirements
====================================================================

To use Genesis `1.4.3 <https://github.com/Genesis-Embodied-AI/genesis-world/releases/tag/v1.4.3>`__, you need the following prerequisites:

- **ROCm version:** `10.1 <https://rocm.docs.amd.com/en/docs-10.1.0/>`__
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI300X, MI325X and MI355X
- **Python:** `3.12 <https://www.python.org/downloads/release/python-31215/>`__
- **PyTorch:** `2.13 <https://github.com/pytorch/pytorch/tree/release/2.13>`__

Install Genesis
====================================================================

This branch is the ROCm 10.1.0 release of Genesis World. The image builds Quadrants from
`AMD-Ecosystem/quadrants rocm10.1.0_r26.10 <https://github.com/AMD-Ecosystem/quadrants/tree/rocm10.1.0_r26.10>`__,
then installs this Genesis tree on that wheel. The base image is ``rocm/dev-ubuntu-24.04:10.1.0-full``.
The host kernel driver must match ROCm 10.1.

Build the Docker image and run the container
--------------------------------------------------------------------

Clone this branch and build from the repository root.

1. Clone the Genesis repository:

   .. code-block:: bash

      git clone -b rocm10.1.0_r26.10 https://github.com/AMD-Ecosystem/Genesis.git
      cd Genesis

2. Build the Docker image:

   .. code-block:: bash

      docker build -t genesis-release-rocm10.1 .

3. Run the container on the AMD GPU. On a machine that also has a display GPU, set all three
   device variables to the CDNA device. That is GPU 0 on the machine used for this release.

   .. code-block:: bash

      docker run --rm -it \
        --device=/dev/kfd --device=/dev/dri \
        --group-add "$(stat -c '%g' /dev/kfd)" \
        -e HIP_VISIBLE_DEVICES=0 \
        -e ROCR_VISIBLE_DEVICES=0 \
        -e CUDA_VISIBLE_DEVICES=0 \
        genesis-release-rocm10.1 \
        bash

      # Print the versions of installed components
      python -c "import torch, quadrants, genesis; print('torch', torch.__version__); print('quadrants', quadrants.__version_str__); print('genesis', genesis.__version__)"

   .. note::

      Do not set ``QD_AMDGPU_V520``.

Test Genesis
====================================================================

After starting ``genesis-release-rocm10.1`` with ``bash``, run these commands inside the container.

Confirm the environment
--------------------------------------------------------------------

Run the following PyTest commands inside the container to test your Genesis environment.

Run the validated Genesis benchmarks
--------------------------------------------------------------------

.. code-block:: bash

   pytest -n 0 -m benchmarks tests/benchmarks/test_rigid.py \
     -k dex_hand \
     --tb=line \
     --speed-test-filepath=/tmp/dex_hand_speed.txt \
     -o cache_dir=/tmp/pytest-cache

Both tests should pass without reporting:

.. code-block:: text

   Invalid constraint forces causing 'nan'

Run all default rigid benchmarks
--------------------------------------------------------------------

.. code-block:: bash

   pytest -n 0 -m benchmarks tests/benchmarks/test_rigid.py \
     --tb=line \
     -o cache_dir=/tmp/pytest-cache

The default ndarray selection includes:

- ``dex_hand``, GPU, batch 4096
- ``g1_fall_accessors``, GPU, batch 4096
- ``table_bussing``, CPU, batch 50

Run the complete Genesis test directory
--------------------------------------------------------------------

.. code-block:: bash

   pytest -n 0 tests \
     --tb=short \
     -o cache_dir=/tmp/pytest-cache

This is substantially longer than the validated benchmark run and can include tests requiring
downloaded assets or optional features. For release verification, ``dex_hand`` and
``g1_fall_accessors`` are the important AMD CDNA checks.
