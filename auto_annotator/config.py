import logging
import os

# ------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("auto_annotator")

# ------------------- GroundingDINO Paths -------------------
ENV_CONFIG = "GROUNDING_DINO_CONFIG_PATH"
ENV_CHECKPOINT = "GROUNDING_DINO_CHECKPOINT_PATH"


def resolve_gdino_paths(config_path=None, checkpoint_path=None):
    """
    Resolve GroundingDINO config and checkpoint paths.
    Priority: explicit args -> environment variables.
    """
    config_path = config_path or os.getenv(ENV_CONFIG)
    checkpoint_path = checkpoint_path or os.getenv(ENV_CHECKPOINT)

    missing = []
    if not config_path:
        missing.append(ENV_CONFIG)
    if not checkpoint_path:
        missing.append(ENV_CHECKPOINT)

    if missing:
        raise FileNotFoundError(
            "Missing GroundingDINO paths. Set env vars: " + ", ".join(missing)
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint_path}")

    return config_path, checkpoint_path
