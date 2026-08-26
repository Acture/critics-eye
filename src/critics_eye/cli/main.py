import logging

import typer

from .primary import primary_app
from .secondary import secondary_app
from .utils import util_app

logger = logging.getLogger(__file__)

app = typer.Typer(help="Critics Eye CLI")
app.add_typer(primary_app)
app.add_typer(secondary_app)
app.add_typer(util_app)

if __name__ == "__main__":
	app()
