from asterinis.neural import EntityMetrics
from asterinis.nlp import NativeSequenceTagger, TokenPrediction
from asterinis.nlp.providers import NeuralNERProvider


def test_native_neural_provider_uses_common_nlp_result() -> None:
    model = NativeSequenceTagger()
    model.predict = lambda text: (
        TokenPrediction("Berlin", "B-LOC", 0.9, 0, 6),
        TokenPrediction("is", "O", 0.8, 7, 9),
        TokenPrediction("nice", "O", 0.8, 10, 14),
    )
    result = NeuralNERProvider(model).analyze("Berlin is nice")

    assert result.entities[0].label == "LOC"
    assert result.entities[0].text == "Berlin"
    assert result.entities[0].start == 0
    assert result.metadata["provider"] == "asterinis-neural-ner"


def test_entity_metrics_is_a_public_native_api() -> None:
    metrics = EntityMetrics(8, 2, 1, 0.8, 8 / 9, 0.8)

    assert metrics.true_positives == 8
    assert metrics.f1 == 0.8
