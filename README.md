
# Asterinis

**Asterinis** is a Python framework for AI orchestration, retrieval-augmented generation (RAG), agent systems, routing, evaluation, confidence-aware decisions, NLP integration, and LLM services.

The framework is designed to bring these components into one backend architecture while keeping them independent, replaceable, and testable.

Asterinis is not tied to one model, NLP library, retriever, vector database, or LLM provider.

---

## Overview

Asterinis provides the coordination layer between the main parts of a modern AI backend.

| Area             | Role in Asterinis                                               |
| ---------------- | --------------------------------------------------------------- |
| **NLP**          | Linguistic analysis and NLP provider integration                |
| **RAG**          | Retrieval, ranking, filtering, scoring, and context preparation |
| **Agents**       | Specialized task execution and coordination                     |
| **LLMs**         | Generation and reasoning providers                              |
| **Routing**      | Chooses the appropriate processing path                         |
| **Confidence**   | Uses confidence values in explicit decision policies            |
| **Evaluation**   | Measures retrieval quality and system behaviour                 |
| **Integrations** | Connects external frameworks and services                       |

These parts can be used together or independently.

---

## Installation

Install Asterinis from PyPI:

```bash
pip install asterinis
```

Optional Flair support:

```bash
pip install "asterinis[flair]"
```

---

## Basic Usage

The main orchestration interface is `Nexus`.

```python
from asterinis import Nexus

nexus = Nexus()

result = nexus.process(
    "Search documents about multilingual NLP."
)

print(result.to_dict())
```

---

## Nexus

`Nexus` is the central interface for connecting routes, providers, agents, and retrieval components.

It is responsible for orchestration rather than performing every task itself.

This keeps individual providers independent from the rest of the application.

---

## Routing

Asterinis includes an extensible routing layer that can decide whether a request should be handled by:

| Route   | Typical use                          |
| ------- | ------------------------------------ |
| `nlp`   | NLP analysis                         |
| `rag`   | Retrieval and document-based queries |
| `agent` | Specialized agent workflows          |
| `llm`   | Direct language-model processing     |

Routing rules can be replaced or extended by the application.

---

## NLP and Flair

Asterinis can integrate **Flair** as an optional NLP provider.

Flair can provide linguistic information such as named entities, classification labels, embeddings, confidence scores, and entity links. Asterinis can then use those signals inside retrieval, routing, agent, and evaluation workflows.

Flair is one possible NLP provider inside Asterinis. The framework is designed to support additional NLP providers as well.

---

## RAG

Asterinis includes a modular retrieval architecture.

Current retrievers include:

| Retriever                | Purpose                                            |
| ------------------------ | -------------------------------------------------- |
| **InMemoryRetriever**    | Lightweight lexical retrieval                      |
| **VectorRetriever**      | Semantic retrieval using embeddings                |
| **HybridRetriever**      | Combines multiple retrieval methods                |
| **EntityAwareRetriever** | Uses entity information during ranking             |
| **FlairRetriever**       | Adds Flair-derived entity information to retrieval |
| **CustomRetriever**      | Connects custom retrieval implementations          |

Asterinis also includes support for retrieval queries, filtering, reranking, scoring, and context building.

---

## RAG Pipeline

The RAG layer contains components that can be combined into a complete retrieval pipeline.

| Component         | Responsibility                                   |
| ----------------- | ------------------------------------------------ |
| `RetrievalQuery`  | Represents a retrieval request                   |
| `RetrievalResult` | Stores retrieved documents and scores            |
| `Retriever`       | Defines the retrieval interface                  |
| `Reranker`        | Reorders retrieval results                       |
| `ContextBuilder`  | Prepares retrieved content for downstream models |
| `RAGPipeline`     | Coordinates the retrieval process                |

The LLM is not hard-wired into the RAG layer, allowing different generation providers to be used.

---

## Agents

Asterinis includes a modular agent system for applications that need specialized processing.

| Component            | Responsibility                       |
| -------------------- | ------------------------------------ |
| **AgentManager**     | Stores and manages available agents  |
| **AgentRouter**      | Selects the appropriate agent        |
| **AgentCoordinator** | Executes the selected agent          |
| **AgentTask**        | Represents work assigned to an agent |
| **AgentContext**     | Carries contextual information       |
| **AgentMemory**      | Stores lightweight state             |
| **Tool**             | Provides reusable capabilities       |
| **AgentResult**      | Standardizes agent output            |

---

## Specialized Agents

Asterinis currently includes several specialized agents.

### ConfidenceAgent

Uses confidence values to make explicit decisions such as:

* `accept`
* `verify`
* `reject`

This allows model confidence to affect system behaviour rather than being used only as metadata.

### EntityRetrievalAgent

Combines entity extraction with retrieval.

It can use entities detected by Flair or another NLP provider to improve the retrieval request.

### RetrievalQualityAgent

Evaluates the quality of retrieved evidence before downstream generation.

Possible decisions include:

* `generate`
* `retrieve_again`
* `clarify`

This provides a control layer between retrieval and generation.

---

## Providers

Providers give Asterinis a common interface for external models, libraries, APIs, and services.

A provider may represent:

* an NLP framework
* an LLM
* a local model
* a retrieval backend
* an external API
* a custom processing service

The provider architecture keeps vendor-specific logic outside the core framework.

---

## Integrations

Asterinis integrations connect external technologies without making them mandatory dependencies.

Current integrations include:

* Flair
* OpenAI

Additional integrations can be added through the provider and connector interfaces.

---

## Confidence-Aware Decisions

Asterinis treats confidence as something that can influence application behaviour.

Instead of simply returning a confidence score, the framework can use configurable policies to decide whether a result should be trusted, verified, rejected, or processed differently.

This is useful for NLP predictions, retrieval results, routing, and agent decisions.

---

## Evaluation

The evaluation layer is intended to make system behaviour measurable.

Current and planned evaluation areas include:

* retrieval quality
* confidence
* ranking
* routing behaviour
* agent decisions
* trace information
* comparison between retrieval strategies

The goal is to make Asterinis useful not only for applications, but also for experimentation and research.

---

## Why Asterinis?

Many AI applications already have access to strong NLP models, retrievers, and LLMs.

The difficult part is often the logic between them.

Asterinis focuses on questions such as:

* Which component should handle this request?
* Should retrieval be used?
* Which retriever should be selected?
* Is the NLP prediction reliable enough?
* Should an agent verify the result?
* Is the retrieved evidence strong enough?
* Should the system generate an answer, retry, or ask for clarification?
* Which provider should be used for the task?

Asterinis keeps these decisions explicit and testable instead of hiding all orchestration inside prompts.

---

## Research Direction

Asterinis is also designed as an environment for experimenting with NLP, retrieval, agents, and LLM systems.

Current areas of interest include:

* entity-aware RAG
* Flair-assisted retrieval
* hybrid retrieval
* confidence-aware routing
* retrieval quality control
* explainable agent decisions
* adaptive orchestration
* comparison of explicit policies with LLM-only decisions

One central research question is whether structured NLP signals can improve retrieval, routing, and downstream decision-making in modern AI systems.

---

## Project Structure

```text
src/
└── asterinis/
    ├── nexus.py
    ├── router.py
    ├── registry.py
    ├── providers.py
    ├── connectors.py
    ├── context.py
    ├── pipeline.py
    ├── hooks.py
    ├── middleware.py
    ├── result.py
    ├── config.py
    ├── exceptions.py
    ├── lifecycle.py
    ├── logging.py
    ├── validators.py
    ├── types.py
    ├── utils.py
    │
    ├── integrations/
    │   ├── flair.py
    │   └── openai.py
    │
    ├── rag/
    │   ├── base.py
    │   ├── documents.py
    │   ├── memory.py
    │   ├── vector.py
    │   ├── hybrid.py
    │   ├── entity.py
    │   ├── flair.py
    │   ├── custom.py
    │   ├── query.py
    │   ├── filters.py
    │   ├── reranker.py
    │   ├── context_builder.py
    │   ├── pipeline.py
    │   └── scores.py
    │
    ├── agents/
    │   ├── base.py
    │   ├── manager.py
    │   ├── router.py
    │   ├── coordinator.py
    │   ├── task.py
    │   ├── result.py
    │   ├── context.py
    │   ├── memory.py
    │   ├── tool.py
    │   ├── confidence.py
    │   ├── entity_retrieval.py
    │   └── retrieval_quality.py
    │
    └── evaluation/
        ├── metrics.py
        └── traces.py
```

---

## Roadmap

Planned development includes:

* additional specialized agents
* stronger RAG evaluation
* more NLP integrations
* persistent memory
* asynchronous orchestration
* additional LLM providers
* richer explainability
* tracing and diagnostics
* benchmark experiments
* adaptive routing policies

---

## Version

Current public release:

```text
0.0.1
```

The next release expands the RAG, provider, and agent architecture.

---

## License

Asterinis is released under the MIT License.

See the [LICENSE](LICENSE) file for details.

---

## Author

Created and maintained by **Irena Popova**, Full-Stack Developer and researcher in AI, NLP, RAG, agent systems, and adaptive software.
