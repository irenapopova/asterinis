from asterinis.agents import NLPRouterAgent


def main() -> None:
    router = NLPRouterAgent(
        default_route="llm"
    )

    router.register(
        "rag",
        lambda text, metadata: bool(
            metadata.get("entities")
        ),
        reason="The request contains entities that can support retrieval.",
        priority=10,
    )

    decision = router.decide(
        "Tell me about Berlin.",
        nlp_metadata={
            "entities": ["Berlin"],
            "confidence": 0.96,
        },
    )

    print(decision.to_dict())


if __name__ == "__main__":
    main()