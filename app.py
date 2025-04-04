from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

from political_sentiment import ElectionPredictor, PoliticalTrendAnalyzer

app = Flask(__name__)

# Mock Sentiment Analyzer for demo purposes
class MockSentimentAnalyzer:
    def predict_sentiment(self, texts):
        return ['positive' if 'great' in text else 'negative' for text in texts]
    
    def preprocess_text(self, text):
        return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Sample data generation
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-04-01', freq='D')
    candidates = ['Smith', 'Johnson']
    topics = ['economy', 'healthcare', 'education', 'immigration']
    
    tweet_templates = [
        "I think {} is doing a great job on {}. #election2024",
        "I don't support {}'s stance on {}. Terrible policy! #politics",
        "{} just announced a new plan for {}. What do you think?",
        "Can't believe what {} said about {} today. #news",
        "The latest poll shows {} is leading on {} issues."
    ]
    
    data = []
    for _ in range(1000):
        date = np.random.choice(dates)
        candidate = np.random.choice(candidates)
        topic = np.random.choice(topics)
        template = np.random.choice(tweet_templates)
        tweet = template.format(candidate, topic)
        sentiment = 'positive' if 'great' in tweet else 'negative'
        retweets = int(np.random.exponential(scale=10))
        data.append({
            'date': date,
            'text': tweet,
            'candidate': candidate,
            'topic': topic,
            'sentiment': sentiment,
            'retweets': retweets
        })
    
    df = pd.DataFrame(data)
    sentiment_analyzer = MockSentimentAnalyzer()
    
    # Analyze trends
    trend_analyzer = PoliticalTrendAnalyzer(sentiment_analyzer)
    sentiment_over_time = trend_analyzer.analyze_temporal_trends(df, 'text', 'date', 'week')
    topic_sentiment = trend_analyzer.analyze_topic_sentiment(df, 'text', topics)
    
    # Predict election
    election_predictor = ElectionPredictor(sentiment_analyzer)
    candidate_keywords = {
        'Smith': ['smith', '#smith', 'smith2024'],
        'Johnson': ['johnson', '#johnson', 'johnson2024']
    }
    issue_keywords = {
        'economy': ['economy', 'economic', 'jobs', 'taxes', 'inflation'],
        'healthcare': ['healthcare', 'health', 'medical', 'insurance'],
        'education': ['education', 'schools', 'students', 'teachers'],
        'immigration': ['immigration', 'border', 'migrants']
    }
    
    features = election_predictor.extract_features(df, 'text', candidate_keywords, issue_keywords)
    predictions = election_predictor.predict_election_outcome(features)
    
    # Convert plots to base64
    def fig_to_base64(fig):
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()
    
    trend_img = fig_to_base64(trend_analyzer.visualize_sentiment_trends(sentiment_over_time))
    topic_img = fig_to_base64(trend_analyzer.visualize_topic_sentiment(topic_sentiment))
    election_img = fig_to_base64(election_predictor.visualize_predictions(predictions))

    return render_template('results.html',
                           sentiment_over_time=sentiment_over_time.to_html(classes='table table-striped'),
                           topic_sentiment=topic_sentiment,
                           predictions=predictions,
                           trend_viz=trend_img,
                           topic_viz=topic_img,
                           election_viz=election_img)

if __name__ == '__main__':
    app.run(debug=True)
