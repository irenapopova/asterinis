from collections.abc import Callable

from .documents import RetrievalResult


ResultFilter = Callable[[RetrievalResult], bool]


def filter_results(
    results: list[RetrievalResult],
    predicate: ResultFilter,
) -> list[RetrievalResult]:
    return [
        result
        for result in results
        if predicate(result)
    ]


def minimum_score(
    threshold: float,
) -> ResultFilter:
    return lambda result: result.score >= threshold