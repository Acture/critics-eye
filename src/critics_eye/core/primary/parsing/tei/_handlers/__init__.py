from .atomic import dispatch_children, handle_atom, handle_speaker, skip
from .header import extract_text, handle_list_person, handle_person, handle_person_group, handle_rs
from .speech import handle_speech
from .stage import handle_keyword, handle_seg, handle_stage
from .structural import handle_div1, handle_scene

from .._models import ElementHandler

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
}

BODY_HANDLERS: dict[str, ElementHandler] = {
	# Structural
	"div1": handle_div1,
	"div2": handle_scene,
	# Content
	"sp": handle_speech,
	"stage": handle_stage,
	# Structured content
	"seg": handle_seg,
	"milestone": skip,
	"q": dispatch_children,
	"ab": dispatch_children,
	"p": dispatch_children,
	"speaker": handle_speaker,
	# Atomic
	"w": handle_atom,
	"c": handle_atom,
	"pc": handle_atom,
	# Keywords
	"sound": handle_keyword,
	"title": handle_keyword,
	"name": handle_keyword,
	"foreign": handle_keyword,
	# Transparent containers
	"hi": dispatch_children,
	# Skip
	"head": skip,
	"lb": skip,
	"pb": skip,
	"fw": skip,
	"app": skip,
	"rdg": skip,
	"gap": skip,
	"label": skip,
	"anchor": skip,
}
