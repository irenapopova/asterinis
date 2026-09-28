import pytest

from asterinis.integrations import FlairTrainingConfig
from asterinis.nlp import (
    Annotation,
    TrainingCorpus,
    TrainingRunConfig,
    TrainingSample,
    TrainingTask,
)


def test_flair_training_config_validates_mode_and_output() -> None:
    config = FlairTrainingConfig(output_path="models/ner", mode="fine_tune")
    assert config.mode == "fine_tune"

    with pytest.raises(ValueError):
        FlairTrainingConfig(output_path="", mode="train")

    with pytest.raises(ValueError):
        FlairTrainingConfig(output_path="models/ner", mode="invalid")


def test_asterinis_training_contract_is_framework_neutral() -> None:
    corpus = TrainingCorpus(
        train=(
            TrainingSample(
                "Berlin is in Germany.",
                annotations=(Annotation("LOC", 0, 6),),
            ),
        )
    )
    config = TrainingRunConfig(
        task=TrainingTask.SEQUENCE_TAGGING,
        output_path="models/ner",
        fine_tune=True,
    )

    assert corpus.labels() == ("LOC",)
    assert config.task == "sequence_tagging"
