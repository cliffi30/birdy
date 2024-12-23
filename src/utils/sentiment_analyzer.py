import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Sentiment analysis model using the Twitter XLM-RoBERTa model
class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

    def analyze_sentiment(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        scores = outputs[0][0].detach().numpy()
        scores = torch.nn.functional.softmax(torch.tensor(scores), dim=0)
        labels = ['negative', 'neutral', 'positive']

        # Get the predicted sentiment and the corresponding score
        sentiment = labels[scores.argmax()]
        sentiment_score = scores.max().item()
        return {"label": sentiment, "score": sentiment_score}