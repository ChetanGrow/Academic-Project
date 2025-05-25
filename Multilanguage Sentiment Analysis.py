import spacy
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Setup
nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

def analyze_text(text):
    print("\n--- Real-Time Text Analysis ---")
    print(f"Input: {text}\n")

    # spaCy NLP pipeline
    doc = nlp(text)

    # Tokenization (alphabetic only)
    tokens = [token.text.lower() for token in doc if token.is_alpha]

    # POS Tagging
    pos_tags = [(token.text, token.pos_) for token in doc]

    # Named Entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Stopword removal
    filtered_tokens = [token for token in tokens if not nlp.vocab[token].is_stop]

    # Word Frequencies
    word_freq = Counter(filtered_tokens).most_common(5)

    # Real-Time Sentiment (VADER)
    scores = sia.polarity_scores(text)
    sentiment = (
        "Positive" if scores["compound"] > 0.05 else
        "Negative" if scores["compound"] < -0.05 else
        "Neutral"
    )

    # Output
    print("Top Words:", word_freq)
    print("POS Tags:", pos_tags)
    print("Named Entities:", entities)
    print(f"Sentiment: {sentiment} (compound score: {scores['compound']})")

# Run the analyzer
if __name__ == "__main__":
    sample_text = input("Enter some text: ")
    analyze_text(sample_text)
