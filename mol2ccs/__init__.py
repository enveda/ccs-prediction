import logging

from mol2ccs.constants import TENSORBOARD_LOG_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("mol2ccs package is being imported.")

# create tensorboard log directory
TENSORBOARD_LOG_DIR.mkdir(exist_ok=True, parents=True)
logger.info(f"Tensorboard log directory: {TENSORBOARD_LOG_DIR.resolve()}")
