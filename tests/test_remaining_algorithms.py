from asterinis.education import CodeSimilarityAnalyzer, TeachingStrategyBandit


def test_strategy_bandit_learns_from_feedback() -> None:
    bandit = TeachingStrategyBandit(("hint", "explain"), exploration=0.0, seed=1)
    selection = bandit.choose(("beginner", "loops"))
    bandit.record(selection, reward=1.0)

    assert bandit.average_reward(selection.strategy, ("beginner", "loops")) == 1.0


def test_code_similarity_is_static_and_detects_structure() -> None:
    analyzer = CodeSimilarityAnalyzer()
    result = analyzer.compare(
        "def add(a, b):\n    return a + b\n",
        "def sum_values(x, y):\n    return x + y\n",
    )

    assert result.valid_first and result.valid_second
    assert result.same_structure
    assert analyzer.compare("not valid(", "x = 1").valid_first is False
