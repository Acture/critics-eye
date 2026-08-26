from pathlib import Path

from networkx import MultiDiGraph

from .parsing.tei import parse
from .parsing.tei_dracor import parse as parse_dracor
from ..models.documents import PrimaryParserType


def load_primary_data(
	p: Path, parser_type: PrimaryParserType = PrimaryParserType.TEI
) -> MultiDiGraph:
	match parser_type:
		case PrimaryParserType.TEI:
			return parse(p)
		case PrimaryParserType.DRACOR:
			return parse_dracor(p)
		case _:
			raise NotImplementedError(f"Parser type {parser_type} not implemented yet.")
