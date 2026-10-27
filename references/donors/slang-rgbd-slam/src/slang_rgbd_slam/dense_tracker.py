"""Fused Caspar residual/Jacobian accumulation for dense ICP and photometric tracking."""

from pathlib import Path

import numpy as np

from .tracker import VulkanTracker


class FusedVulkanTracker(VulkanTracker):
    def __init__(
        self,
        camera,
        *,
        photo_weight=0.03,
        photo_huber=0.1,
        group_size=64,
        reduction="subgroup",
        **options,
    ):
        from caspar_slang.reprojection import dense_factor_source

        if (
            not np.isfinite([photo_weight, photo_huber]).all()
            or photo_weight < 0
            or photo_huber <= 0
        ):
            raise ValueError("invalid photometric weight or Huber threshold")
        if group_size not in (32, 64, 128, 256) or reduction not in ("subgroup", "workgroup"):
            raise ValueError("invalid dense reduction specialization")
        super().__init__(camera, _compile_association=False, _reduction="workgroup", **options)
        self.photo_weight, self.photo_huber = photo_weight, photo_huber
        minimum_wave = getattr(self.runtime.device.info.limits, "min_wave_size", 1)
        minimum_wave = max(1, int(minimum_wave or 1))
        directory = Path(__file__).with_name("shaders")
        source = (
            f"#define DENSE_GROUP_SIZE {group_size}\n#define DENSE_USE_WAVE {int(reduction == 'subgroup')}\n#define DENSE_MIN_WAVE_SIZE {minimum_wave}\n"
            + dense_factor_source()
            + "\n"
            + directory.joinpath("tracker.slang").read_text()
            + "\n"
            + directory.joinpath("dense_tracker.slang").read_text()
        )
        self.fused = self.runtime.compile(source, entry="fusedAssociate")
        self.gray = self.runtime.compile(source, entry="rgbIntensity")
        self.gray_downsample = self.runtime.compile(source, entry="downsampleIntensity")
        for level in self.levels:
            # Release original correspondence and residual/Jacobian storage.
            level["problem"].buffers.clear()
            if group_size != 64:
                count = (level["problem"].count + group_size - 1) // group_size
                partials = []
                while True:
                    partials.append((count, self.runtime.buffer(np.zeros(count * 43, np.float32))))
                    if count == 1:
                        break
                    count = (count + 63) // 64
                level["problem"].partials = partials
            for role in ("source", "reference"):
                level["maps"][role]["intensity"] = self.runtime.buffer(
                    np.zeros(level["problem"].count, np.float32)
                )

    def record_intensity(self, rgb_buffer, encoder):
        level = self.levels[0]
        self.gray.dispatch(
            thread_count=[level["problem"].count, 1, 1],
            vars=dict(
                rgbImage=rgb_buffer,
                intensityOutput=level["maps"]["source"]["intensity"],
                mapParams=self._map_params(level["camera"]),
            ),
            command_encoder=encoder,
        )

    def _record_intensity_pyramid(self, role, encoder):
        for i, level in enumerate(self.levels[1:], 1):
            encoder.global_barrier()
            self.gray_downsample.dispatch(
                thread_count=[level["camera"].width, level["camera"].height, 1],
                vars=dict(
                    sourceIntensity=self.levels[i - 1]["maps"][role]["intensity"],
                    intensityOutput=level["maps"][role]["intensity"],
                    mapParams=self._map_params(level["camera"]),
                ),
                command_encoder=encoder,
            )

    def _record_track(self, command_encoder):
        self._record_maps("source", command_encoder)
        command_encoder.global_barrier()
        self._record_intensity_pyramid("source", command_encoder)
        command_encoder.global_barrier()
        self._record_intensity_pyramid("reference", command_encoder)
        command_encoder.global_barrier()
        for level, iterations in zip(reversed(self.levels), self.iterations):
            camera, problem = level["camera"], level["problem"]
            params = dict(
                count=problem.count,
                width=camera.width,
                height=camera.height,
                fx=camera.fx,
                fy=camera.fy,
                cx=camera.cx,
                cy=camera.cy,
                distance=self.distance,
                normalCos=self.normal_cos,
                huber=self.huber,
                minCorrespondences=self.minimum,
            )
            source, reference = level["maps"]["source"], level["maps"]["reference"]
            for _ in range(iterations):
                self.kernels["resetCount"].dispatch(
                    thread_count=[1, 1, 1],
                    vars=dict(trackingStatus=self.status),
                    command_encoder=command_encoder,
                )
                command_encoder.global_barrier()
                self.fused.dispatch(
                    thread_count=[problem.count, 1, 1],
                    vars=dict(
                        sourceVertices=source["vertices"],
                        sourceNormals=source["normals"],
                        referenceVertices=reference["vertices"],
                        referenceNormals=reference["normals"],
                        sourceIntensity=source["intensity"],
                        referenceIntensity=reference["intensity"],
                        pose=self.pose,
                        trackingStatus=self.status,
                        outputTerms=problem.partials[0][1],
                        trackParams=params,
                        photoParams=dict(weight=self.photo_weight, huber=self.photo_huber),
                    ),
                    command_encoder=command_encoder,
                )
                command_encoder.global_barrier()
                for (count, previous), (_, output) in zip(problem.partials, problem.partials[1:]):
                    self.gn.reduce_partials.dispatch(
                        thread_count=[count, 1, 1],
                        vars=dict(
                            inputTerms=previous, outputTerms=output, params=dict(count=count)
                        ),
                        command_encoder=command_encoder,
                    )
                    command_encoder.global_barrier()
                self.gn.solve.dispatch(
                    thread_count=[1, 1, 1],
                    vars=dict(
                        inputTerms=problem.partials[-1][1],
                        delta=problem.delta,
                        status=problem.status,
                        solveParams=dict(damping=self.damping),
                    ),
                    command_encoder=command_encoder,
                )
                command_encoder.global_barrier()
                self.kernels["updatePose"].dispatch(
                    thread_count=[1, 1, 1],
                    vars=dict(
                        pose=self.pose,
                        initialPose=self.initial_pose,
                        delta=problem.delta,
                        solveStatus=problem.status,
                        normalTerms=problem.partials[-1][1],
                        trackingStatus=self.status,
                        diagnostics=self.diagnostics,
                        trackParams=params,
                    ),
                    command_encoder=command_encoder,
                )
