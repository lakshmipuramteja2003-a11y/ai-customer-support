# AI Customer Support Assistant

A Streamlit chat app that uses LangChain to answer e-commerce support questions with Ollama or OpenRouter. Conversation history is kept in the active Streamlit session. The assistant has no connection to live orders and cannot issue refunds or change accounts.

## Setup

From the workspace root, create and activate the project virtual environment, then install dependencies:

```bash
cd ai_customer_support
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows, activate with `.venv\Scripts\activate` instead.

### Ollama

Install Ollama. If its service does not start automatically (as in this Codespace), start it in a separate terminal and leave it running:

```bash
ollama serve
```

Then pull the configured model:

```bash
ollama pull llama3.2
```

Adjust `OLLAMA_MODEL` and `OLLAMA_BASE_URL` in `.env` if needed.

### OpenRouter

Add your key to `OPENROUTER_API_KEY` in `.env`, then set `LLM_PROVIDER=openrouter`. Set `OPENROUTER_MODEL` to a model available to your OpenRouter account. Keep `.env` private.

## Run

```bash
cd ai_customer_support
./.venv/bin/streamlit run app.py
```

You can switch provider and model from the sidebar. The provider configured in `.env` is the initial selection.

## Manual evaluation

Try a delayed delivery, an item that arrived damaged, a return request, and a refund follow-up. Check that answers are relevant and consistent with earlier turns, ask for missing details, and do not claim to have accessed an order or completed an account action. Store-specific policies and order facts must be supplied by the customer; this starter app does not have a commerce-system integration.