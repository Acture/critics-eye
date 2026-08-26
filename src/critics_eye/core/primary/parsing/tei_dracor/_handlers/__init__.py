from critics_eye.core.primary.parsing.tei._handlers.header import (
	extract_text,
	handle_list_person,
	handle_person,
	handle_person_group,
	handle_rs,
)

from .._models import ElementHandler
from .speech import handle_line, handle_line_group, handle_speech, skip_speaker
from .stage import handle_stage
from .structural import dispatch_children, handle_div, skip

HEADER_HANDLERS: dict[str, ElementHandler] = {
	"listPerson": handle_list_person,
	"person": handle_person,
	"personGrp": handle_person_group,
	"persName": extract_text,
	"sex": extract_text,
	"state": extract_text,
	"death": extract_text,
	"name": extract_text,
	"p": extract_text,
	"rs": handle_rs,
	"head": skip,
	"lb": skip,
	"roleName": extract_text,
	"title": extract_text,
	"listRelation": skip,
	"relation": skip,
}

BODY_HANDLERS: dict[str, ElementHandler] = {
	"div": handle_div,
	"sp": handle_speech,
	"stage": handle_stage,
	"l": handle_line,
	"p": handle_line,
	"lg": handle_line_group,
	"speaker": skip_speaker,
	"hi": dispatch_children,
	"emph": dispatch_children,
	"foreign": dispatch_children,
	"q": dispatch_children,
	"seg": dispatch_children,
	"ab": dispatch_children,
	"pb": skip,
	"head": skip,
	"trailer": skip,
}
