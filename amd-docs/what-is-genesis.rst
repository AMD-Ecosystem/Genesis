.. meta::
  :description: What is Genesis?
  :keywords: Genesis, documentation, simulation, robotics, physical AI, framework, GPU, AMD, ROCm, overview, introduction

.. _what-is-genesis:

********************************************************************
What is Genesis?
********************************************************************

Genesis World, still often called Genesis, is a simulation platform for physical AI.
It combines a unified multi-physics engine, camera rendering, and a cross-platform
compiler behind a Python scene API. The same scene description can drive rigid robots,
deformable materials, fluids, and coupled solvers, then render what those solvers produce.

The compiler underneath the numeric kernels is Quadrants. Upstream Genesis is developed
as genesis-world. This ROCm edition is the
`rocm10.1.0_r26.10 <https://github.com/AMD-Ecosystem/Genesis/tree/rocm10.1.0_r26.10>`__
branch of `AMD-Ecosystem/Genesis <https://github.com/AMD-Ecosystem/Genesis>`__.

Why Genesis?
====================================================================

Genesis is aimed at robotics and embodied-AI workloads that need a lot of simulated experience:

- **One scene, several solvers:** Rigid bodies, FEM, MPM, particle methods (PBD and SPH),
  an IPC solver, and an explicit coupler share a scene and a state instead of living in
  separate simulators.

- **A Python interface over compiled kernels:** Assets, entities, controllers, and sensors
  are ordinary Python. The inner loops are Quadrants kernels, so a laptop script and a
  batched GPU run use the same scene code.

- **Batching for learning:** Parallel and heterogeneous environments, domain randomization,
  and sensors (cameras, depth, IMU, lidar, contact) are part of the simulation interface,
  which is the shape robot-learning and data-generation pipelines expect.

- **A renderer in the same stack:** Camera sensors can use Nyx, the Luisa ray tracer, or
  Pyrender, so perception data can be produced from the same scene as the physics steps.

Features and use cases
====================================================================

Genesis provides the following features:

- **Physics:** Rigid contact and articulation, FEM, MPM, SPH, PBD (cloth and liquid),
  stable fluids, IPC through the uipc backend, SAP coupling, and couplers that attach
  those solvers to one another.

- **Assets and control:** URDF, MJCF, OBJ, GLB, USD, and related formats; joint
  controllers, differential IK, and batched IK.

- **Sensors and viewing:** Depth, IMU, lidar, tactile, contact force, and a built-in
  GUI for joint control and debug drawing.

- **Parallel environments:** Many copies of a scene on one device, including
  heterogeneous environments and domain randomization.

- **Rendering:** Nyx for robotics-oriented cameras, Luisa for ray tracing, and
  Pyrender for rasterization.

Genesis is commonly used in the following scenarios:

- **Robot learning:** Train or evaluate policies against articulated robots such as
  manipulators, humanoids, and quadrupeds, with batched environments on the GPU.

- **Synthetic data:** Generate camera, depth, and contact traces from scripted or
  randomized scenes.

- **Multi-physics experiments:** Study contact, soft bodies, cloth, and fluids in one
  process instead of stitching simulators together.

- **Compiler bring-up:** Use the rigid benchmarks as a check that Quadrants is producing
  correct, fast kernels on a new GPU backend.
