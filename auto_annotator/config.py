import logging

# ------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("gdino")

# ------------------- Hardcoded Config & Checkpoint -------------------
GROUNDING_DINO_CONFIG_PATH = "C:\\Wraith\\auto_annotate\\GroundingDINO\\groundingdino\\config\\GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "C:\\Wraith\\auto_annotate\\GroundingDINO\\weights\\groundingdino_swint_ogc.pth"
