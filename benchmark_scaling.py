#!/usr/bin/env python3
"""
Clean version of benchmark_scaling.py with proper timing control separation
"""

import argparse
import time
import json
import os
from typing import List, Optional

import genesis as gs

# Import the timing controller
from timing_control import configure_timing_from_args, timing_controller


def run_benchmark(n_envs, num_steps=500, precision="64", timing_mode="disabled"):
    """Run a benchmark with clean timing control"""

    gs.init(backend=gs.amdgpu, precision=precision)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.005,
            constraint_solver=gs.constraint_solver.CG,
            iterations=15,
            tolerance=1e-6,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.URDF(
            file="./newton-assets/unitree_g1/urdf/g1_29dof.urdf",
            pos=(0, 0, 1.0),
        ),
    )

    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    # Configure timing on the rigid solver
    rigid_solver = scene.sim.rigid_solver

    if timing_mode == "detailed":
        timing_controller.enable_detailed_timing()
        rigid_solver.enable_debug_timing(
            enable_instrumented_kernels=False,
            enable_detailed_timing=True,
            enable_forward_dynamics_breakdown=True,  # Always enable breakdown in detailed mode
            enable_kernel_step_1_instrumented=False
        )
        print(f"[INFO] Detailed timing with full breakdown enabled for {n_envs} environments")

    elif timing_mode == "basic":
        timing_controller.enable_basic_timing()
        rigid_solver.enable_debug_timing(
            enable_instrumented_kernels=False,
            enable_detailed_timing=False,
            enable_forward_dynamics_breakdown=False,
            enable_kernel_step_1_instrumented=False
        )
        print(f"[INFO] Basic timing enabled for {n_envs} environments")

    else:  # disabled
        timing_controller.disable_timing()
        rigid_solver.disable_debug_timing()
        print(f"[INFO] All timing disabled - maximum performance mode")

    # Warmup
    warmup_steps = 10
    print(f"[INFO] Warmup ({warmup_steps} steps)...")
    for i in range(warmup_steps):
        scene.step()

    # Timed run
    start = time.perf_counter()
    for step in range(num_steps):
        scene.step()
    elapsed = time.perf_counter() - start

    # Print debug statistics only if enabled
    if timing_controller.is_basic_timing_enabled:
        print(f"\n[TIMING] Timing statistics enabled")
        rigid_solver.print_debug_stats()

    fps = num_steps / elapsed
    total_fps = fps * n_envs

    print(f"n_envs={n_envs:>5d}  |  wall_time={elapsed:.2f}s  |  FPS={fps:.1f}  |  throughput={total_fps:.0f} env·steps/s")

    # Clean up
    scene.destroy()
    gs.destroy()

    return {
        "n_envs": n_envs,
        "wall_time_s": round(elapsed, 3),
        "fps": round(fps, 2),
        "throughput": round(total_fps, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Genesis AMD GPU scaling benchmark with clean timing control")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--max-envs", type=int, default=8192)
    parser.add_argument("--n-envs", type=str, default=None)
    parser.add_argument("--precision", type=str, default="64", choices=["32", "64"])

    # Unified timing control
    parser.add_argument("--timing", type=str, default="disabled",
                       choices=["disabled", "basic", "detailed"],
                       help="Timing mode: disabled (fastest), basic (kernel timing), detailed (full breakdown)")

    # Backward compatibility flags
    parser.add_argument("--enable-debug-timing", action="store_true",
                       help="Enable basic timing (same as --timing basic)")
    parser.add_argument("--enable-detailed-timing", action="store_true",
                       help="Enable detailed timing with full breakdown (same as --timing detailed)")

    args = parser.parse_args()

    # Configure timing mode
    timing_mode = args.timing

    # Handle backward compatibility
    if args.enable_detailed_timing:
        timing_mode = "detailed"
    elif args.enable_debug_timing:
        timing_mode = "basic"

    # Parse environment counts
    if args.n_envs:
        env_counts = [int(x.strip()) for x in args.n_envs.split(",")]
    else:
        env_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        env_counts = [n for n in env_counts if n <= args.max_envs]

    print(f"Benchmark: {args.num_steps} steps, precision={args.precision}, timing={timing_mode}")
    print(f"Env counts: {env_counts}")
    print("=" * 80)

    results = []
    for i, n_envs in enumerate(env_counts):
        print(f"\n[{i+1}/{len(env_counts)}] Running benchmark with {n_envs} environments...")
        result = run_benchmark(
            n_envs,
            num_steps=args.num_steps,
            precision=args.precision,
            timing_mode=timing_mode
        )
        results.append(result)

    # Save results
    os.makedirs("benchmark_results", exist_ok=True)
    suffix = f"_fp{args.precision}" if args.precision != "64" else ""
    suffix += f"_{timing_mode}" if timing_mode != "disabled" else ""
    results_path = f"benchmark_results/scaling_results{suffix}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()