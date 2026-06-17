# Prompt Optimizer — OpenAI API
#
# A terminal tool that analyses a raw prompt and rewrites it using the most
# appropriate prompting technique: Few-shot, Chain of Thought, or ReAct.
#
# Model  : gpt-4o-mini
# Pricing: $0.60 / 1M prompt tokens | $0.15 / 1M completion tokens

from openai import OpenAI
import os
from dotenv import load_dotenv

# --- 1. Setup ---
# Load API key from .env and initialise the OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 2. System Prompt ---
# Defines the optimizer's behaviour and the three rewriting patterns
system_prompt = {
    "role": "system",
    "content": """You are a prompt optimisation tool. Your job is to improve raw prompts provided by the user.

Follow this process for every input:

STEP 1 — Analysis
Analyse the user's prompt and the task it describes.

STEP 2 — Category Selection
Identify the most suitable prompting technique:
- Few-shot    — use when you need to control output format. The model must always produce
                the same structure. Provide examples to demonstrate it.
- Chain of Thought — use when the model must reason before answering. Best for complex
                     tasks, analysis, debugging, and multi-step decisions.
- ReAct       — use when the model needs external data to answer (API calls, web search,
                file reading). Do not use this category if no tools are available.

STEP 3 — Quality Check
Evaluate the original prompt against these examples:
- Example 1 — prompt: "Build me a website"
  Verdict: Too vague, not suitable — missing context, role, technology and scope.
- Example 2 — prompt: "You are a web development expert for a cotton manufacturing company.
  Use HTML, CSS and JavaScript. Build a simple homepage for the user."
  Verdict: Suitable — needs only minor refinements.

STEP 4 — Rewrite
Rewrite the prompt following the pattern of the identified category:

Few-shot (three variants):
  -System role, and the problem that client ask to resolve
  
  - Zero-shot — instruction only, no examples
  - One-shot  — one example
  - Few-shot  — two or more examples

Chain of Thought:
  -System role, and the problem that client ask to resolve
  
  [Task instruction]
  Think step by step.
  [Input]

ReAct:
  -System role, and the problem that client ask to resolve
  
  Thought: [reasoning]
  Action: [what to do]
  Observation: [result of the action]
  Thought: [updated reasoning]
  Action: [next action]
  ...
  Answer: [final answer]

Reason step by step at every stage — this is mandatory.

OUTPUT FORMAT:

2. Improved prompt
3. Explanation of the changes made"""
}

# --- 3. Optimizer Loop ---
while True:

    user_input = input("User: ")

    # Exit on quit command
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye.")
        break

    # Send system prompt + user message to the API
    # No conversation memory needed — each prompt is analysed independently
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            system_prompt,
            {"role": "user", "content": user_input}
        ]
    )

    print("\nAssistant:", response.choices[0].message.content, "\n")
