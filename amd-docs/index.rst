.. meta::
  :description: Genesis documentation
  :keywords: Genesis, ROCm, documentation, simulation, robotics, physical AI, framework, GPU

.. _genesis-documentation-index:

********************************************************************
Genesis on ROCm documentation
********************************************************************

Genesis World, still often called Genesis, is a simulation platform for physical AI.
It combines a unified multi-physics engine, camera rendering, and a cross-platform
compiler behind a Python scene API. The compiler underneath the numeric kernels is
Quadrants. Upstream Genesis is developed as genesis-world. This ROCm edition is the
`rocm10.1.0_r26.10 <https://github.com/AMD-Ecosystem/Genesis/tree/rocm10.1.0_r26.10>`__
branch of AMD-Ecosystem/Genesis.

Use Genesis on ROCm to simulate physical AI workloads on AMD Instinct GPUs.
The same Python scene can step rigid robots, deformable materials, fluids, and
coupled solvers, then render what those solvers produce.

The Genesis public repository is located at
`https://github.com/AMD-Ecosystem/Genesis <https://github.com/AMD-Ecosystem/Genesis>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install Genesis <install/genesis-install>`

  .. grid-item-card:: Reference

      * `Genesis World documentation (upstream) <https://genesis-world.readthedocs.io/en/latest/>`__
      * `Quadrants documentation (upstream) <https://genesis-embodied-ai.github.io/quadrants/index.html>`__

To contribute to the documentation, refer to
`Contributing to Genesis <https://github.com/AMD-Ecosystem/Genesis/blob/docs/26.10/.github/contributing/PULL_REQUESTS.md>`__.

You can find licensing information on the :doc:`Licensing <about/license>` page.
