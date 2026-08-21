# asterinis
# Asterinis

**Asterinis** is an early-stage Python framework for connecting NLP, retrieval-augmented generation (RAG), AI agents, routing logic, explainability, and LLM services through a unified backend nexus.

The project is designed as a lightweight orchestration layer that can connect different AI and NLP components without locking an application to a single model provider or framework.

## Current Status

Asterinis is currently in an early development stage.

The first public release focuses on the core framework structure, including:

* Nexus orchestration
* Rule-based routing
* Connector interfaces
* Processing context
* Pipelines
* Hooks
* Result objects
* Optional NLP integrations

More advanced RAG, agent orchestration, adaptive routing, evaluation, and explainability features are planned for future releases.

## Installation

The package will be installable from PyPI:

```bash
pip install asterinis
```

For optional Flair NLP support:

```bash
pip install "asterinis[flair]"
```

## Basic Usage

```python
from asterinis import Nexus

nexus = Nexus()

result = nexus.process(
    "Search documents about multilingual NLP."
)

print(result.route)
print(result.payload)
```

Asterinis can route requests toward different processing paths such as:

```text
rag
nlp
agent
llm
```

## Flair Integration

Asterinis supports Flair as an optional NLP connector.

```python
from asterinis import Nexus
from asterinis.integrations import FlairConnector

nexus = Nexus()

nexus.register_connector(
    "nlp",
    FlairConnector("ner")
)

result = nexus.process(
    "Deutsche Bank is based in Frankfurt."
)

print(result.to_dict())
```

Flair is an independent open-source NLP framework and is not part of Asterinis. Asterinis provides an integration layer that allows Flair models to participate in a broader orchestration pipeline.

## Architecture

```text
Application
    ↓
Asterinis Nexus
    ↓
Router
    ↓
┌──────────┬──────────┬──────────┬──────────┐
│   NLP    │   RAG    │  Agents  │   LLM    │
└──────────┴──────────┴──────────┴──────────┘
    ↓
Connectors / Integrations
```

The goal is to allow developers to plug different AI systems into a consistent interface.

## Project Structure

```text
src/
└── asterinis/
    ├── __init__.py
    ├── nexus.py
    ├── router.py
    ├── connectors.py
    ├── context.py
    ├── pipeline.py
    ├── hooks.py
    ├── result.py
    ├── config.py
    ├── exceptions.py
    └── integrations/
        ├── __init__.py
        └── flair.py
```

## Roadmap

Future development may include:

* advanced RAG routing
* retrieval quality evaluation
* agent orchestration
* adaptive routing policies
* explainability
* confidence-aware decision logic
* additional NLP integrations
* LLM provider connectors
* evaluation tools
* persistent context and memory
* asynchronous pipelines

## Version

Current development version:

```text
0.0.1
```

## License

Asterinis is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Created and maintained by **Irena Popova**, Full-Stack Developer and AI/NLP, RAG, and adaptive systems researcher.