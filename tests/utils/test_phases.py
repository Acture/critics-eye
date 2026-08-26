from critics_eye.core.secondary.relation import LitAnalysis
from critics_eye.utils.text import build_vocabulary_and_contexts, extract_phrases


def test_extract_phases_against_review(mac_reviews: list[LitAnalysis]):
	for review in mac_reviews:
		phases = extract_phrases(review.justification_quote)
		print(review.source_file, phases)


def test_extract_phases_against_act(Mac_text: str):
	phases = extract_phrases(Mac_text)
	print(phases)


def test_build_vocabulary_and_contexts(Mac_text: str, mac_reviews: list[LitAnalysis]):
	c, con = build_vocabulary_and_contexts(Mac_text, mac_reviews)
	print(c, con)


def test_build_cooccurrence_matrix(Mac_text: str, mac_reviews: list[LitAnalysis]):
	keywords, contexts = build_vocabulary_and_contexts(Mac_text, mac_reviews)
	print(keywords)
