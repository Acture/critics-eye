from pathlib import Path

from networkx import MultiDiGraph

from ._handlers import BODY_HANDLERS, HEADER_HANDLERS
from ._parser import DraCorTEIParser

_default_parser = DraCorTEIParser(
	body_handlers=BODY_HANDLERS,
	header_handlers=HEADER_HANDLERS,
)


def parse(p: Path) -> MultiDiGraph:
	return _default_parser.parse(p)
