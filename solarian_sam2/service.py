from solarian.services import Service

from solarian_sam2.processor import Sam2BboxProcessor

#: torch.compile's caches, in the image rather than on the volume: the image
#: build's warm-up fills them, on the GPU type the endpoint serves on.
COMPILE_CACHE_DIR = "/opt/torch-compile-cache"


class Sam2TrackerService(Service):
    """SAM 2 camera predictor: seed on a clicked point, then follow the object."""

    server_module = "solarian_sam2.server"
    default_port = 8765
    processor_cls = Sam2BboxProcessor

    @classmethod
    def modal_spec(cls):
        from solarian.core.ai.system1.policy_trainer import ModalSpec

        # sam2 (the segment-anything-2-real-time checkout installed here) ships
        # as source: pip-installing it compiles its CUDA kernel, which needs a
        # CUDA toolkit the build container lacks. Without the kernel SAM 2 skips
        # filling small mask holes, as it does on a Mac.
        # The sam2.1_hiera_*.pt weights are not baked in: the server downloads
        # them into ./checkpoints, which modal-serve keeps on the volume.
        # Served --compiled, the build's warm-up compiles the model, and a
        # container starts in seconds instead of minutes.
        return ModalSpec(
            apt_packages=["libgl1", "libglib2.0-0"],
            pip_packages=["torch", "torchvision", "fastapi", "uvicorn"],
            local_python_packages=["solarian_sam2", "sam2"],
            env={
                "TORCHINDUCTOR_CACHE_DIR": COMPILE_CACHE_DIR,
                "TRITON_CACHE_DIR": f"{COMPILE_CACHE_DIR}/triton",
                "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
                "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
            },
            bake_warm_up=True,
        )
