from critics_eye.utils.text import split_into_sentences


def test_split_into_sentences():
	res = list(split_into_sentences("This is a test.LOL."))
	assert res == ["This is a test.", "LOL."]


def test_pdf_texts(secondary_pdf_texts):
	for text in secondary_pdf_texts:
		sentences = split_into_sentences(text)
		assert sentences
