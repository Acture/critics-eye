import itertools
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, Dict, Generator, List, Set

import numpy as np
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from spacy import Language
from tqdm import tqdm

from critics_eye.core.secondary.relation.model import LitAnalysis


def split_into_sentences(
	raw_text: str, lang: str = "en", splitter: str = "\n\n"
) -> Generator[str, Any, None]:
	"""Splits a text into sentences."""
	_LANG_TO_SPACY_MAP = {
		"zh": "zh_core_web_sm",
		"en": "en_core_web_sm",
	}
	if lang not in _LANG_TO_SPACY_MAP:
		raise ValueError(f"Unsupported language: {lang}")

	nlp: Language = spacy.load(_LANG_TO_SPACY_MAP[lang])

	if len(raw_text) >= nlp.max_length:
		logging.warning(
			f"Text length {len(raw_text)} exceeds model max length {nlp.max_length}. Splitting..."
		)

		text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=nlp.max_length, chunk_overlap=0
		)
		texts = text_splitter.split_text(raw_text)

	else:
		texts = [raw_text]

	for text in texts:
		if not text:
			logging.warning("Empty text found in texts. Skipping...")
			continue
		nlp_text = nlp(text)
		for sent in nlp_text.sents:
			yield sent.text_with_ws


def normalize_text(text: str) -> str:
	if not text:
		return ""
	# collapse \n, \r, tabs
	text = re.sub(r"[\r\n\t]+", " ", text)
	# collapse multiple spaces
	text = re.sub(r"\s{2,}", " ", text)
	# remove surrounding quotes
	text = text.strip(' "“”‘’')
	# unify dashes
	text = text.replace("—", "-").replace("–", "-")
	return text.strip()


tei_words: set[str] = {
	"act",
	"play",
	"scene",
	"title",
	"speaker",
	"stage",
	"enter",
	"exit",
}


def is_content_phrase(span_doc):
	# 只留下至少有一个 content word 的短语
	return any(
		tok.pos_ in {"NOUN", "PROPN", "ADJ"}
		and not tok.is_stop
		and tok.text not in tei_words
		for tok in span_doc
	)


def clean_text(raw: str) -> str:
	if raw is None:
		return ""

	text = raw.lower()

	# 删除 PDFDrive 及书名中的 metadata
	text = re.sub(r"\(.*?pdfdrive.*?\)", " ", text)
	text = re.sub(r"\(.*?modern critical interpretations.*?\)", " ", text)

	# 删除剧本行号（1.3.130, 3.2.24 等）
	text = re.sub(r"\b\d+\.\d+\.\d+\b", " ", text)
	text = re.sub(r"\b\d+\.\d+\b", " ", text)

	# 删除超过 3 位的数字（年份/页码）
	text = re.sub(r"\b\d{3,}\b", " ", text)

	# 删除换行符
	text = text.replace("\n", " ").replace("\r", " ")

	# 删除重复空格
	text = re.sub(r"\s+", " ", text)

	return text.strip()


def is_character_like(span_text: str, character_lexicon: set[str]) -> bool:
	if character_lexicon is None:
		return False
	t = span_text.strip().lower()
	if not t:
		return False
	# 完全等于某个角色名或其 token
	if t in character_lexicon:
		return True
	# 短语中含有角色 token（lady macbeth, the witches, his wife...）
	tokens = t.split()
	if any(tok in character_lexicon for tok in tokens):
		return True
	return False


SHAKESPEARE_STOP = {
	"hath",
	"thou",
	"thee",
	"thy",
	"thine",
	"doth",
	"ere",
	"aye",
	"nay",
	"o'",
	"ne’er",
	"’tis",
	"tis",
	"’twas",
}

# 泛化无信息名词（必须过滤）
GENERIC_NOUNS = {
	"hand",
	"heart",
	"eyes",
	"eye",
	"face",
	"head",
	"light",
	"night",
	"day",
	"man",
	"woman",
	"thing",
	"way",
	"place",
	"world",
	"life",
	"time",
	"word",
	"mind",
	"thought",
	"look",
	"sound",
	"voice",
	"moment",
	"house",
	"door",
	"room",
	"lord",
	"sir",
	"lady",
	"king",
	"queen",
	"soldier",
	"master",
	"child",
}

# 舞台方向词（TEI 已表示，不属于 motif）
STAGE_WORDS = {"alarum", "hautboy", "exeunt", "exit", "enter"}


def is_bad_noun(s: str) -> bool:
	return s in GENERIC_NOUNS or s in SHAKESPEARE_STOP or s in STAGE_WORDS


def extract_phrases(
	text: str,
	nlp: Language | None = None,
	character_lexicon: set[str] | None = None,
	min_len: int = 3,
) -> dict[str, int]:
	if nlp is None:
		nlp = spacy.load("en_core_web_sm", disable=["ner"])

	text = normalize_text(text)
	text = clean_text(text)
	if not text:
		return {}

	doc = nlp(text)
	phrases = Counter()

	# --- Helper filters ---
	def ok_token(tok):
		if tok.is_stop or tok.is_space or tok.is_punct:
			return False
		t = tok.lemma_.lower()
		if len(t) < min_len:
			return False
		if is_bad_noun(t):
			return False
		if character_lexicon and is_character_like(t, character_lexicon):
			return False
		return True

	# --- 1) Multi-word noun chunks ---
	for chunk in doc.noun_chunks:
		span = chunk.text.lower().strip()
		# 过滤太短、不含字母、角色名、垃圾 noun-chunk
		if len(span) < min_len:
			continue

		if any(is_bad_noun(tok.lemma_.lower()) for tok in chunk):
			continue

		# 去除 “my hand”, “your eye”, “the light” 这种无 motif 价值的结构
		if chunk.root.pos_ == "NOUN" and is_bad_noun(chunk.root.lemma_.lower()):
			continue

		if character_lexicon and is_character_like(span, character_lexicon):
			continue

		if not is_content_phrase(chunk):
			continue

		phrases[span] += 1

	# --- 2) Single-word content nouns ---
	for tok in doc:
		if not ok_token(tok):
			continue

		lemma = tok.lemma_.lower().strip()
		# 避免加入 TEI 已识别词，如 Macbeth, Banquo, Duncan…
		if character_lexicon and lemma in character_lexicon:
			continue

		phrases[lemma] += 1

	return phrases


def build_vocabulary_and_contexts(
	play_text: str,
	reviews: list[LitAnalysis],
	nlp: Language | None = None,
	min_freq: int = 3,
	max_keywords: int = 2000,
	character_lexicon: set[str] | None = None,
) -> tuple[List[str], List[Set[str]]]:
	freq = defaultdict(int)
	review_phrase_sets: List[Set[str]] = []

	if nlp is None:
		nlp: Language = spacy.load("en_core_web_sm", disable=["ner"])

	for review in reviews:
		review_pharse = extract_phrases(
			review.source_element, nlp, character_lexicon=character_lexicon
		)
		for k, v in review_pharse.items():
			freq[k] += v
		review_pharse = extract_phrases(
			review.target_element, nlp, character_lexicon=character_lexicon
		)
		for k, v in review_pharse.items():
			freq[k] += v
		review_phrase_sets.append(set(review_pharse.keys()))

	candidates = [phrase for phrase, c in freq.items() if c >= min_freq]

	candidates.sort(key=lambda p: (-freq[p], p))

	if max_keywords is not None and len(candidates) > max_keywords:
		candidates = candidates[:max_keywords]

	logging.info(
		f"[INFO] #raw phrases = {len(freq)}, " f"#filtered keywords = {len(candidates)}"
	)

	return candidates, review_phrase_sets


def build_cooccurrence_matrix(
	phrases: List[str],
	review_phrase_sets: List[Set[str]],
) -> np.ndarray:
	"""
	对每条 review 的短语集合，统计 phrase-phrase 共现次数。
	返回 cooc 矩阵 shape = [N, N]，对角线 = 0。
	"""
	vocab_index: Dict[str, int] = {p: i for i, p in enumerate(phrases)}
	N = len(phrases)
	cooc = np.zeros((N, N), dtype=np.float32)

	for phrases_in_review in tqdm(review_phrase_sets, desc="Building cooccurrence"):
		# 只保留在 vocab 里的短语
		indices = [vocab_index[p] for p in phrases_in_review if p in vocab_index]
		if len(indices) < 2:
			continue

		# 对所有 unordered pair 加 1（可选：用 1/comb(len,2) 做归一）
		for i, j in itertools.combinations(indices, 2):
			cooc[i, j] += 1.0
			cooc[j, i] += 1.0

	# 保留 cooc 对角线为 0
	np.fill_diagonal(cooc, 0.0)

	logging.info(f"Non-zero cooc entries: {np.count_nonzero(cooc)}")
	return cooc


def top_k_global_pairs(C: np.ndarray, phrases, k=50):
	N = C.shape[0]
	pairs = []

	for i, j in itertools.combinations(range(N), 2):
		if C[i, j] > 0:
			pairs.append((C[i, j], i, j))

	pairs.sort(reverse=True)  # 从最大次数开始

	logging.info(f"Top {k} co-occurring pairs in reviews:\n")
	for count, i, j in pairs[:k]:
		logging.info(f"{count:4f}  {phrases[i]}   <-->   {phrases[j]}")


def compute_embeddings(phrases: Sequence[str], model_name: str) -> np.ndarray:
	logging.info(f"[INFO] Loading embedding model: {model_name}")
	model = SentenceTransformer(model_name)
	embeddings = model.encode(
		list(phrases),
		batch_size=64,
		show_progress_bar=True,
		convert_to_numpy=True,
		normalize_embeddings=True,  # 方便后面用点积当 cosine
	)
	logging.info(f"[INFO] Embeddings shape = {embeddings.shape}")
	return embeddings
