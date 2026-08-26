from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from rich.progress import (
	BarColumn,
	MofNCompleteColumn,
	Progress,
	SpinnerColumn,
	TextColumn,
	TimeElapsedColumn,
	TimeRemainingColumn,
	track,
)

T = TypeVar("T")


def new_progress(*, transient: bool = False) -> Progress:
	return Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
		BarColumn(),
		TextColumn("{task.percentage:>3.0f}%"),
		MofNCompleteColumn(),
		TimeElapsedColumn(),
		TimeRemainingColumn(),
		transient=transient,
	)


def iter_with_progress(
	iterable: Iterable[T],
	*,
	description: str,
	total: int | None = None,
	transient: bool = False,
) -> Iterable[T]:
	return track(
		iterable,
		description=description,
		total=total,
		transient=transient,
	)


def batch_count(
	item_count: int,
	batch_size: int,
	*,
	minimum_batch_size: int = 1,
) -> int:
	if batch_size <= 0:
		raise ValueError(f"batch_size must be positive; got {batch_size}.")
	if minimum_batch_size <= 0:
		raise ValueError(
			f"minimum_batch_size must be positive; got {minimum_batch_size}."
		)

	count = 0
	for start in range(0, item_count, batch_size):
		end = min(start + batch_size, item_count)
		if end - start >= minimum_batch_size:
			count += 1
	return count
