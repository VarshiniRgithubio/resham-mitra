# 0001 - Phase 1 stack choices

**Qdrant over Chroma / pgvector**: hybrid dense + sparse search helps Kannada, where exact terms (disease and chemical names) matter and multilingual embeddings are weaker. Payload filters (language, source) make experiment E2 clean. pgvector would add Postgres next to MongoDB.

**MongoDB for app data**: chats, feedback and expert ratings are document-shaped and change shape often during research.

**LLM behind one interface (`app/llm/base.py`)**: experiments must swap models without code changes. A `fake` provider makes tests run offline and free.

**Health endpoint first**: proves the services are wired before any RAG logic exists, and gives a quick check on any new machine.

Interview one-liner: "I separated the model from the pipeline so I can compare models and safety methods fairly, with config as the only variable."
