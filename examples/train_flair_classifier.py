"""Train a Flair classifier through Asterinis.

Install first:
    pip install "asterinis[flair]"
"""

from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier

from asterinis.integrations import FlairTrainerAdapter, FlairTrainingConfig


corpus: Corpus = TREC_6()
label_type = "question_class"
labels = FlairTrainerAdapter.label_dictionary(corpus, label_type)
embeddings = TransformerDocumentEmbeddings(
    "distilbert-base-uncased",
    fine_tune=True,
)
model = TextClassifier(
    embeddings,
    label_dictionary=labels,
    label_type=label_type,
)

result = FlairTrainerAdapter().train(
    model,
    corpus,
    config=FlairTrainingConfig(
        output_path="models/question-classifier",
        mode="fine_tune",
        learning_rate=5e-5,
        batch_size=4,
        epochs=10,
    ),
)
print(result)
