# AI Sentiment Analyzer

A smart web-based Sentiment Analysis project developed using **Python**, **Flask**, **NLTK**, **spaCy**, and **Transformers**.
This application analyzes user-entered text and predicts whether the sentiment is **Positive**, **Negative**, or **Neutral** with a confidence score.

## Features

* Detects sentiment from any sentence
* Predicts:

  * Positive 😊
  * Negative 😞
  * Neutral 😐
* Confidence percentage display
* Attractive modern UI
* Search history feature
* Dashboard analytics with charts
* Uses advanced NLP models

## Technologies Used

* Python
* Flask
* NLTK
* spaCy
* Hugging Face Transformers
* HTML
* CSS
* JavaScript
* Chart.js

## Project Structure

```text
AI-Sentiment-Analyzer/
│── app.py
│── requirements.txt
│── templates/
│   └── index.html
│── static/   (optional)
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ai-sentiment-analyzer.git
cd ai-sentiment-analyzer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download spaCy model:

```bash
python -m spacy download en_core_web_sm
```

4. Run the project:

```bash
python app.py
```

5. Open browser:

```text
http://127.0.0.1:5000
```

## Example Inputs

* I love this phone
* Worst experience ever
* Today is normal
* The movie was average
* I got selected in interview

## Future Enhancements

* Multilingual sentiment analysis
* Voice input support
* Emotion detection
* User login system
* Cloud deployment

## Author

Developed by Shiva Sumedha

## License

This project is for educational and learning purposes.
