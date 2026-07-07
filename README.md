# Finance AI Engineer Portfolio

Production-grade AI engineering projects built for financial services use cases.

## Projects

### Email Intelligence Pipeline
Multi-agent LangGraph workflow for email summarization and action extraction.
- Supervisor + worker agent architecture
- Conditional routing (simple vs complex emails)
- Shared state management across agents
- Built with LangChain Anthropic + LangGraph

**Stack:** Python, LangGraph, LangChain, Claude Haiku

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 email_intelligence_pipeline.py
```
