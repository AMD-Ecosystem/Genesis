import argparse
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--num-robots", type=int, default=1)
    parser.add_argument("--no-render", action="store_true", help="Disable rendering/recording")
    parser.add_argument("--collision-multiplier", type=int, default=1, help="Collision broad phase multiplier")
    args = parser.parse_args()

    ########################## init ##########################
    # Genesis 0.3.14 uses gs.gpu (Taichi supports AMD via ROCm)
    gs.init(backend=gs.amdgpu, precision="64")
    ########################## create a scene ##########################
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            multiplier_collision_broad_phase=args.collision_multiplier,  # Configurable via CLI
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=False,  # Headless mode for server without display
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    # Load Unitree G1 robot (will be replicated across all num_envs)
    g1_robot = scene.add_entity(
        gs.morphs.URDF(file="/root/Genesis/newton-assets/unitree_g1/urdf/g1_29dof.urdf"),
        visualize_contact=True,
    )

    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        res=(1280, 960),
        pos=(3.5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=30,
        GUI=True,
    )

    ########################## build ##########################
    scene.build(n_envs=args.num_robots, env_spacing=(1.0, 1.0))

    # Start recording if rendering is enabled
    if not args.no_render:
        cam_0.start_recording()

    # Run simulation
    for i in range(args.num_steps):
        scene.step()

        if not args.no_render:
            cam_0.render()

    # Save recording if rendering was enabled
    if not args.no_render:
        os.makedirs("benchmark_results/video", exist_ok=True)
        video_filename = f"benchmark_results/video/g1_robots_{args.num_robots}_steps_{args.num_steps}.mp4"
        cam_0.stop_recording(save_to_filename=video_filename, fps=50)
        print(f"Recording saved to: {video_filename}")


if __name__ == "__main__":
    main()
