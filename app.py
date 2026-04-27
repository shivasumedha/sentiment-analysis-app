from flask import Flask, render_template, request
from transformers import pipeline
import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

app = Flask(__name__)

# Load Models
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Store History
history = []

# Dashboard Counts
positive_count = 0
negative_count = 0
neutral_count = 0

def analyze_sentiment(text):
    global positive_count, negative_count, neutral_count

    doc = nlp(text)

    pred = classifier(text)[0]
    tf_label = pred["label"]
    tf_score = pred["score"]

    vader = sia.polarity_scores(text)
    compound = vader["compound"]

    if -0.15 < compound < 0.15:
        result = "Neutral"
        emoji = "😐"
        confidence = 78.5
        neutral_count += 1

    else:
        if tf_label == "POSITIVE":
            result = "Positive"
            emoji = "😊"
            positive_count += 1
        else:
            result = "Negative"
            emoji = "😞"
            negative_count += 1

        confidence = round(tf_score * 100, 2)

    history.insert(0, f"{text} → {result}")

    if len(history) > 5:
        history.pop()

    return result, emoji, confidence

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = ""
    emoji = ""
    sentence = ""

    if request.method == "POST":
        sentence = request.form.get("sentence", "").strip()

        if sentence:
            result, emoji, confidence = analyze_sentiment(sentence)
            confidence = str(confidence) + "%"

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

if __name__ == "__main__":
    app.run(debug=True)