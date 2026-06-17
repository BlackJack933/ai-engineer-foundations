# AI Engineer Foundations

A hands-on learning path to build practical AI engineering skills from scratch.
Each step produces a working terminal project using real APIs and pre-trained models.

---

## Projects

### Step 1 — OpenAI Chatbot
**File:** `step1-openai-chatbot/chatbot.py`

A conversational chatbot that runs in the terminal using the OpenAI Chat Completions API.
Maintains a sliding memory window of 20 messages to manage context and token costs.
Tracks and displays cumulative API costs at the end of each session.

- **Model:** `gpt-4o-mini`
- **Key concepts:** Chat Completions API, token management, sliding context window, cost tracking

---

### Step 2 — Prompt Optimizer
**File:** `step2-prompt-optimizer/prompt_optimizer.py`

A terminal tool that analyses a raw user prompt and rewrites it using the most appropriate
prompting technique: Few-shot, Chain of Thought, or ReAct.

- **Model:** `gpt-4o-mini`
- **Key concepts:** Few-shot prompting, Chain of Thought, ReAct, system prompt engineering

---

### Step 3 — Sentiment Classifier
**File:** `step3-sentiment-classifier/sentiment_classifier.py`

A terminal-based sentiment analysis tool that classifies user input as POSITIVE or NEGATIVE
with a confidence score. Runs entirely on local hardware using a pre-trained transformer model
— no API key required.

- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Key concepts:** Pre-trained models, transfer learning, Hugging Face Transformers pipeline, local inference

---

## Installation

```bash
pip install openai python-dotenv transformers torch
```

For Step 1 and Step 2, create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

---

## Usage

Run each project from its folder:

```bash
# Step 1
python step1-openai-chatbot/chatbot.py

# Step 2
python step2-prompt-optimizer/prompt_optimizer.py

# Step 3
python step3-sentiment-classifier/sentiment_classifier.py
```

Type `exit` to quit any of the tools.
