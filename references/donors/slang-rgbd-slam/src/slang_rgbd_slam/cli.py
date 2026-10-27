"""Run the Vulkan RGB-D prototype and write reviewable artifacts."""

import argparse
import json
from pathlib import Path


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(prog="slang-rgbd-slam")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="independent ray-rendered sequence with known poses")
    demo.add_argument("--frames", type=positive, default=40)
    demo.add_argument("--width", type=positive, default=160)
    demo.add_argument("--height", type=positive, default=120)
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    run = sub.add_parser("run", help="track a calibrated TUM RGB-D dataset")
    run.add_argument("dataset", type=Path)
    run.add_argument("--preset", choices=["tum-fr3"])
    run.add_argument("--width", type=positive, default=640)
    run.add_argument("--height", type=positive, default=480)
    for name in ["fx", "fy", "cx", "cy"]:
        run.add_argument("--" + name, type=float)
    run.add_argument("--depth-scale", type=float, default=5000.0)
    run.add_argument("--association-tolerance", type=float, default=0.02)
    run.add_argument("--max-frames", type=positive)
    run.add_argument("--output", type=Path, default=Path("outputs/tum"))
    run.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    demo.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    compare = sub.add_parser("compare", help="matched CPU-only versus GPU TSDF pipeline")
    compare.add_argument("--frames", type=positive, default=40)
    compare.add_argument(
        "--dataset", type=Path, help="TUM Freiburg3 calibrated dataset; fixed 640x480 preset"
    )
    compare.add_argument(
        "--quality-views",
        type=positive,
        help="limit evaluation raycasts; every frame is still timed",
    )
    compare.add_argument("--width", type=positive, default=160)
    compare.add_argument("--height", type=positive, default=120)
    compare.add_argument("--output", type=Path, default=Path("outputs/comparison"))
    benchmark = sub.add_parser("benchmark", help="time resident GPU preprocessing and ICP")
    benchmark.add_argument("--width", type=positive, default=160)
    benchmark.add_argument("--height", type=positive, default=120)
    benchmark.add_argument("--iterations", type=positive, default=5)
    for command in (demo, run, compare):
        command.add_argument(
            "--max-blocks",
            type=positive,
            default=8192,
            help="bounded TSDF capacity, at most 32767 blocks",
        )
    for command in (demo, run, compare):
        command.add_argument("--engine", choices=["hybrid", "baseline"], default="hybrid")
        command.add_argument("--max-keyframes", type=positive, default=16)
        command.add_argument("--keyframe-interval", type=positive, default=5)
        command.add_argument("--ba-every-keyframes", type=positive, default=4)
        command.add_argument("--disable-loops", action="store_true")
        command.add_argument("--disable-local-ba", action="store_true")
    tune = sub.add_parser("tune", help="validate and benchmark hardware specializations")
    tune.add_argument(
        "--output", type=Path, default=Path.home() / ".cache/slang-rgbd-slam/tuning.json"
    )
    tune.add_argument("--iterations", type=positive, default=25)
    args = parser.parse_args()
    if args.command == "tune":
        from caspar_slang.runtime import VulkanRuntime

        from .hardware import tune as tune_hardware

        print(
            json.dumps(
                tune_hardware(VulkanRuntime(), args.output, iterations=args.iterations), indent=2
            )
        )
        return 0
    hybrid_options = {}
    if getattr(args, "engine", "baseline") == "hybrid":
        hybrid_options = dict(
            max_keyframes=args.max_keyframes,
            keyframe_interval=args.keyframe_interval,
            ba_every_keyframes=args.ba_every_keyframes,
            enable_loops=not args.disable_loops,
            enable_local_ba=not args.disable_local_ba,
        )
    if args.command == "benchmark":
        from .benchmark import run_benchmark

        print(json.dumps(run_benchmark(args.width, args.height, args.iterations), indent=2))
        return 0
    if args.command == "compare":
        if args.engine == "hybrid":
            from .hybrid_comparison import compare_hybrid_backends as compare_backends
        else:
            from .comparison import compare_backends
        if args.engine == "hybrid":
            from .textured_synthetic import textured_sequence as synthetic_sequence
        else:
            from .synthetic import synthetic_sequence

        if args.dataset:
            from itertools import islice

            from .dataset import TUMDataset
            from .tracker import Camera

            camera = Camera(640, 480, 535.4, 539.2, 320.1, 247.6)
            dataset = TUMDataset(args.dataset, camera)

            class Prefix(list):
                pass

            frames = Prefix(islice(dataset, args.frames))
            frames.ground_truth = dataset.ground_truth
        else:
            camera, frames = synthetic_sequence(args.frames, args.width, args.height)
        print(
            json.dumps(
                compare_backends(
                    frames,
                    camera,
                    args.output,
                    max_evaluation_poses=args.quality_views,
                    map_options={"max_blocks": args.max_blocks},
                    **({"hybrid_options": hybrid_options} if args.engine == "hybrid" else {}),
                ),
                indent=2,
            )
        )
        return 0
    from .slam import run_sequence

    if args.command == "demo":
        if args.engine == "hybrid":
            from .textured_synthetic import textured_sequence as synthetic_sequence
        else:
            from .synthetic import synthetic_sequence

        camera, frames = synthetic_sequence(args.frames, args.width, args.height)
        report = run_sequence(
            frames,
            camera,
            args.output,
            backend=args.backend,
            engine=args.engine,
            **hybrid_options,
            map_options={"max_blocks": args.max_blocks},
        )
        print(json.dumps(report, indent=2))
        return 0 if report["lost_frames"] == 0 and report["mapping_failed_frames"] == 0 else 1
    from .dataset import TUMDataset
    from .tracker import Camera

    supplied = [args.fx, args.fy, args.cx, args.cy]
    if args.preset:
        if args.width != 640 or args.height != 480 or any(v is not None for v in supplied):
            parser.error("tum-fr3 preset requires 640x480 and no explicit intrinsics")
        intrinsics = [535.4, 539.2, 320.1, 247.6]
    elif all(v is not None for v in supplied):
        intrinsics = supplied
    else:
        parser.error("supply --preset tum-fr3 or all of --fx --fy --cx --cy")
    camera = Camera(args.width, args.height, *intrinsics)
    dataset = TUMDataset(
        args.dataset, camera, depth_scale=args.depth_scale, tolerance=args.association_tolerance
    )
    report = run_sequence(
        dataset,
        camera,
        args.output,
        args.max_frames,
        backend=args.backend,
        engine=args.engine,
        **hybrid_options,
        map_options={"max_blocks": args.max_blocks},
    )
    print(json.dumps(report, indent=2))
    return (
        0
        if report["frames"] and report["lost_frames"] == 0 and report["mapping_failed_frames"] == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
