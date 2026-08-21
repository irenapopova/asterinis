# src/asterinis/integrations/flair.py

from typing import Any

from asterinis.connectors import Connector


class FlairConnector(Connector):
    name = "flair"

    def __init__(self, model_name: str = "ner"):
        try:
            from flair.nn import Classifier
        except ImportError as exc:
            raise ImportError(
                "Flair is not installed. "
                'Install it with: pip install "asterinis[flair]"'
            ) from exc

        self.model_name = model_name
        self.model = Classifier.load(model_name)

    def execute(self, payload: Any) -> Any:
        from flair.data import Sentence

        text = payload["text"]

        sentence = Sentence(text)
        self.model.predict(sentence)

        entities = []

        for entity in sentence.get_spans("ner"):
            label = entity.get_label("ner")

            entities.append(
                {
                    "text": entity.text,
                    "label": label.value,
                    "confidence": label.score,
                }
            )

        return {
            "provider": "flair",
            "model": self.model_name,
            "text": text,
            "entities": entities,
        }