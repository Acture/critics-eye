import json
from typing import Literal

from pydantic import BaseModel, model_validator


class ReviewMotifIndex(BaseModel):
	index: int
	review_id: int
	kind: Literal["source", "target"]
	phrase: str

	@staticmethod
	def from_json(json_str: str) -> list["ReviewMotifIndex"]:
		json_data: list[dict] = json.loads(json_str)
		return [ReviewMotifIndex.model_validate(d) for d in json_data]


def _normalize_phrase(phrase: str) -> str:
	return phrase.strip().lower()


class _BaseMotifCluster(BaseModel):
	motif_id: int
	representative_phrase: str
	representative_index: int
	members: list[ReviewMotifIndex]

	@model_validator(mode="after")
	def sync_representative_member(self) -> "_BaseMotifCluster":
		if not self.members:
			raise ValueError("Motif clusters must contain at least one member.")

		if 0 <= self.representative_index < len(self.members):
			member_phrase = self.members[self.representative_index].phrase
			if _normalize_phrase(member_phrase) == _normalize_phrase(
				self.representative_phrase
			):
				self.representative_phrase = member_phrase
				return self

		for idx, member in enumerate(self.members):
			if _normalize_phrase(member.phrase) == _normalize_phrase(
				self.representative_phrase
			):
				self.representative_index = idx
				self.representative_phrase = member.phrase
				return self

		if 0 <= self.representative_index < len(self.members):
			self.representative_phrase = self.members[self.representative_index].phrase
			return self

		self.representative_index = 0
		self.representative_phrase = self.members[0].phrase
		return self

	@property
	def size(self) -> int:
		return len(self.members)

	@property
	def keywords(self) -> set[str]:
		return {m.phrase for m in self.members}

	@property
	def symbol(self) -> str:
		return self.representative_phrase


class MotifCluster(_BaseMotifCluster):

	@staticmethod
	def from_json(json_str: str) -> list["MotifCluster"]:
		json_data: list[dict] = json.loads(json_str)
		return [MotifCluster.model_validate(d) for d in json_data]


class LabeledMotifCluster(_BaseMotifCluster):
	labels: list[str]

	@staticmethod
	def from_json(json_str: str) -> list["LabeledMotifCluster"]:
		json_data: list[dict] = json.loads(json_str)
		return [LabeledMotifCluster.model_validate(d) for d in json_data]
