
# Finance AI Engineer Portfolio

Production-grade AI engineering projects for financial services.

Built to demonstrate Senior AI Engineer capabilities.

## Architecture

## Architecture

```

User Query

     ↓

Classifier Agent (LangGraph)

     ↓

     ├── Compliance → RAG Agent (ChromaDB + Claude)

     ├── Email      → Summarization Agent (Claude Haiku)

     └── General    → LLM Router (Haiku vs Sonnet)

     ↓

LangSmith Monitoring

     ↓

Response + Cost + Latency

```





## Projects

| File | Description | Tech |

|------|-------------|------|

| finance_ai_system.py | Full integrated system | LangGraph + RAG + LLM Router |

| email_intelligence_pipeline.py | Multi-agent email processing | LangGraph + Claude |

| llm_router.py | Dynamic model selection | Claude Haiku + Sonnet |

| rag_pipeline.py | Regulatory document Q&A | ChromaDB + LlamaIndex |

| api_service.py | Production API | FastAPI + Async Python |

| prompt_engineer.py | A/B testing prompts | LangSmith + Claude |

| monitoring.py | Observability | LangSmith tracing |

## Stack

- **LLMs:** Claude Haiku, Claude Sonnet (Anthropic)

- **Orchestration:** LangGraph, LangChain

- **Vector DB:** ChromaDB

- **API:** FastAPI, Uvicorn

- **Monitoring:** LangSmith

- **Container:** Docker

- **Language:** Python 3.9

## Setup

```bash

python3 -m venv venv

source venv/bin/activate

pip3 install -r requirements.txt

python3 finance_ai_system.py

```

## Yahoo Context

Built on patterns from Yahoo Mail Intelligence —

5,000 QPS LLM serving with Qwen 1.7B, vLLM,

multi-agent orchestration, and LoRA fine-tuning.


## Live Demo

API deployed on GKE — Google Kubernetes Engine

**Base URL:** http://34.9.87.94

Test it:
```bash
curl http://34.9.87.94/health

curl -X POST http://34.9.87.94/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Basel III capital ratio?"}'
```
