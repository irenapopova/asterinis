import pytest

from asterinis.nlp import (
    Annotation,
    NativeSequenceTagger,
    NativeSequenceTaggerConfig,
    TrainingCorpus,
    TrainingRunConfig,
    TrainingSample,
    TrainingTask,
)


def test_native_sequence_config_and_corpus_are_available_without_torch() -> None:
    config = NativeSequenceTaggerConfig(embedding_dim=8, hidden_dim=8)
    corpus = TrainingCorpus(
        train=(TrainingSample("Berlin", (Annotation("LOC", 0, 6),)),)
    )
    assert config.embedding_dim == 8
    assert corpus.labels() == ("LOC",)


def test_native_tagger_reports_optional_dependency() -> None:
    tagger = NativeSequenceTagger()
    run = TrainingRunConfig(
        task=TrainingTask.SEQUENCE_TAGGING,
        output_path="/tmp/asterinis-test-model.pt",
    )
    corpus = TrainingCorpus(train=(TrainingSample("Berlin"),))
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"asterinis\[neural\]"):
            tagger.fit(corpus, config=run)
