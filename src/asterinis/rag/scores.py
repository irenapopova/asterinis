def normalize_score(score: float) -> float:
    if score < 0:
        return 0.0

    if score > 1:
        return 1.0

    return float(score)


def weighted_score(
    base_score: float,
    *,
    lexical: float = 0.0,
    semantic: float = 0.0,
    entity: float = 0.0,
) -> float:
    return float(
        base_score
        + lexical
        + semantic
        + entity
    )