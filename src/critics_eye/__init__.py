import logging
import os

import dotenv
import torch
from rich.logging import RichHandler
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage, EdgeStorage, NodeStorage

# --- 1. Load Environment FIRST ---
# This loads .env variables into os.environ
dotenv.load_dotenv(verbose=True)

# --- 2. Determine Log Level from Environment ---
LOG_LEVEL_MAP = {
	"DEBUG": logging.DEBUG,
	"INFO": logging.INFO,
	"WARNING": logging.WARNING,
	"ERROR": logging.ERROR,
	"CRITICAL": logging.CRITICAL,
}
# Get the LOG_LEVEL from os.environ, defaulting to "INFO" if not set
env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = LOG_LEVEL_MAP.get(env_level, logging.INFO)

# --- 3. Configure Logging (Now with the correct level) ---
# We set 'force=True' to overwrite any default logging config
logging.basicConfig(
	level=log_level,
	format="%(message)s",
	datefmt="[%X]",
	handlers=[RichHandler(rich_tracebacks=False, markup=True, keywords=["CE"])],
	force=True,
)

# You can now create a logger for your application
log = logging.getLogger("critics_eye")
log.info(f"Logging level set to: [bold cyan]{env_level}[/bold cyan]")

# --- 4. (Optional) Show what dotenv found ---
# Use dotenv_values() *only* for the report, not for loading
env_values = dotenv.dotenv_values()
if env_values:
	# Use log.debug for this, as it's verbose
	log.debug(f"Loaded variables from .env: {list(env_values.keys())}")
else:
	log.debug(".env file not found or is empty.")

# --- Your application code can now use the logger ---
log.debug("This is a DEBUG message.")
log.info("This is an INFO message.")
log.warning("This is a WARNING message.")
