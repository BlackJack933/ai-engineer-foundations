# Sentiment Classifier — Hugging Face Transformers
#
# A terminal-based sentiment analysis tool that classifies user input
# as POSITIVE or NEGATIVE with a confidence score.
#
# Model     : distilbert-base-uncased-finetuned-sst-2-english
# Task      : Sentiment Analysis (binary classification)
# Source    : https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english

from transformers import pipeline

# --- 1. Setup ---
# Load the sentiment analysis pipeline with an explicit model and tokenizer.
# Specifying both prevents Hugging Face from downloading a default model,
# ensuring reproducibility across environments.
sentiment_task = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
)

# --- 2. Inference Loop ---
# Continuously prompt the user for input until they type "exit".
while True:

    user = input("User: ")

    # Exit condition — gracefully terminate the loop on quit command
    if user.lower() == "exit":
        print("Goodbye")
        break

    # Run inference: the pipeline returns a list of dicts, one per input.
    # Each dict contains:
    #   - 'label' : predicted class ("POSITIVE" or "NEGATIVE")
    #   - 'score' : confidence score as a float between 0.0 and 1.0
    risultato = sentiment_task(user)

    # --- 3. Output ---
    # Display the predicted label and confidence score formatted as a percentage.
    # risultato[0] accesses the first (and only) result since we pass a single string.
    print(f"Sentiment: {risultato[0]['label']} ({risultato[0]['score']:.2%})")
