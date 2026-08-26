from aenum import StrEnum, auto


class DocumentType(StrEnum):
	PRIMARY = auto()
	COMMENTARY = auto()


class PrimaryParserType(StrEnum):
	TEI = auto()
	# DraCor ships the same TEI standard but a different dialect of it: no @n on
	# <l>, act and scene numbers spelled out in the heading, choruses hung as
	# bare <lg>/<p>. It needs its own parser, not a flag on the Folger one.
	DRACOR = auto()
