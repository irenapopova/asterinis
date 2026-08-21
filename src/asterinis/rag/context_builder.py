from .documents import RetrievalResult


class ContextBuilder:
    """
    Builds LLM-ready context from retrieved documents.
    """

    def build(
        self,
        results: list[RetrievalResult],
        *,
        include_metadata: bool = False,
    ) -> str:

        blocks: list[str] = []

        for index, result in enumerate(results, start=1):
            document = result.document

            block = (
                f"[Document {index}]\n"
                f"{document.text}"
            )

            if include_metadata and document.metadata:
                block += (
                    f"\nMetadata: {document.metadata}"
                )

            blocks.append(block)

        return "\n\n".join(blocks)