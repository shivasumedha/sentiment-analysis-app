from flask import Flask, render_template, request
from transformers import pipeline
import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

# --------------------------------
# Flask App
# --------------------------------
app = Flask(__name__)

# --------------------------------
# Safe spaCy Load
# --------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")

# --------------------------------
# Safe NLTK Setup
# --------------------------------
try:
    sia = SentimentIntensityAnalyzer()
except:
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()

# --------------------------------
# Load Transformer Model
# --------------------------------
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# --------------------------------
# Store History
# --------------------------------
history = []

positive_count = 0
negative_count = 0
neutral_count = 0

# --------------------------------
# Sentiment Function
# --------------------------------
def analyze_sentiment(text):
    global positive_count, negative_count, neutral_count

    # spaCy processing
    doc = nlp(text)

    # Transformer prediction
    pred = classifier(text)[0]
    label = pred["label"]
    score = pred["score"]

    # NLTK support
    vader = sia.polarity_scores(text)
    compound = vader["compound"]

    if -0.15 < compound < 0.15:
        result = "Neutral"
        emoji = "😐"
        confidence = 78.5
        neutral_count += 1

    else:
        if label == "POSITIVE":
            result = "Positive"
            emoji = "😊"
            positive_count += 1
        else:
            result = "Negative"
            emoji = "😞"
            negative_count += 1

        confidence = round(score * 100, 2)

    # Save History
    history.insert(0, f"{text} → {result}")

    if len(history) > 5:
        history.pop()

    return result, emoji, confidence

# --------------------------------
# Main Route
# --------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = ""
    emoji = ""
    sentence = ""

    if request.method == "POST":
        sentence = request.form.get("sentence", "").strip()

        if sentence:
            try:
                result, emoji, confidence = analyze_sentiment(sentence)
                confidence = str(confidence) + "%"

            except:
                result = "Error"
                emoji = "⚠️"
                confidence = "0%"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        emoji=emoji,
        sentence=sentence,
        history=history,
        positive=positive_count,
        negative=negative_count,
        neutral=neutral_count
    )

# --------------------------------
# Run App
# --------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)