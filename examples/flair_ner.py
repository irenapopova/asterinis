from asterinis.integrations import FlairProvider


def main() -> None:
    provider = FlairProvider("ner")

    result = provider.invoke(
        "Deutsche Bank is based in Frankfurt."
    )

    for entity in result["entities"]:
        print(
            entity["text"],
            entity["label"],
            round(entity["confidence"], 4),
        )


if __name__ == "__main__":
    main()