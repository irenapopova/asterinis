from asterinis import Nexus


def main() -> None:
    nexus = Nexus()

    result = nexus.process(
        "Search documents about multilingual NLP."
    )

    print(result.to_dict())


if __name__ == "__main__":
    main()