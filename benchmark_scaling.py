import argparse
import time
import json
import os

import genesis as gs


def run_benchmark(n_envs, num_steps=500, precision="64"):
    """Run a benchmark with the given number of environments. Returns FPS."""
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
            file="/src/Genesis/newton-assets/unitree_g1/urdf/g1_29dof.urdf",
            pos=(0, 0, 1.0),
        ),
    )

    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    # Warmup (first few steps include JIT compilation)
    for _ in range(10):
        scene.step()

    # Timed run
    start = time.perf_counter()
    for _ in range(num_steps):
        scene.step()
    elapsed = time.perf_counter() - start

    fps = num_steps / elapsed
    total_fps = fps * n_envs  # throughput = steps/sec * envs

    print(f"n_envs={n_envs:>5d}  |  wall_time={elapsed:.2f}s  |  FPS={fps:.1f}  |  throughput={total_fps:.0f} env·steps/s")

    # Clean up for the next run
    scene.destroy()
    gs.destroy()

    return {
        "n_envs": n_envs,
        "wall_time_s": round(elapsed, 3),
        "fps": round(fps, 2),
        "throughput": round(total_fps, 1),
    }


def make_combined_plot():
    """Load fp64 and fp32 results and create a combined plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for precision, color, label, results_path in [
        ("64", "#2196F3", "FP64", "benchmark_results/scaling_results.json"),
        ("32", "#FF5722", "FP32", "benchmark_results/scaling_results_fp32.json"),
    ]:
        if not os.path.exists(results_path):
            continue
        with open(results_path) as f:
            results = json.load(f)

        envs = [r["n_envs"] for r in results]
        throughput = [r["throughput"] for r in results]

        ax.plot(envs, throughput, "o-", linewidth=2, markersize=8, color=color, label=label)

        for r in results:
            ax.annotate(
                f"{r['throughput']:.0f}",
                (r["n_envs"], r["throughput"]),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=7,
                color=color,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Number of Environments", fontsize=13)
    ax.set_ylabel("Throughput (env·steps / sec)", fontsize=13)
    ax.set_title("Genesis G1 Robot – AMD MI300X Scaling\n(CG solver, 15 iters, tol=1e-6, dt=5ms, 500 steps)", fontsize=14)
    ax.set_xticks(envs)
    ax.set_xticklabels([str(e) for e in envs], rotation=45)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plot_path = "benchmark_results/scaling_plot.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Genesis AMD GPU scaling benchmark")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--max-envs", type=int, default=8192)
    parser.add_argument("--n-envs", type=int, nargs="+", help="Exact env count(s) to benchmark (skips sweep)")
    parser.add_argument("--precision", type=str, default="64", choices=["32", "64"])
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate the combined plot")
    args = parser.parse_args()

    if args.plot_only:
        make_combined_plot()
        return

    if args.n_envs:
        env_counts = args.n_envs
    else:
        env_counts = []
        n = 1
        while n <= args.max_envs:
            env_counts.append(n)
            n *= 2

    print(f"Benchmark: {args.num_steps} steps, CG solver (15 iters, tol=1e-6), dt=5ms, precision={args.precision}")
    print(f"Env counts: {env_counts}")
    print("=" * 80)

    results = []
    for n_envs in env_counts:
        result = run_benchmark(n_envs, num_steps=args.num_steps, precision=args.precision)
        results.append(result)

    # Save results
    os.makedirs("benchmark_results", exist_ok=True)
    suffix = f"_fp{args.precision}" if args.precision != "64" else ""
    results_path = f"benchmark_results/scaling_results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Combined plot
    make_combined_plot()


if __name__ == "__main__":
    main()
 
