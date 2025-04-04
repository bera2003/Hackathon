import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class PoliticalSentimentAnalyzer:
    def __init__(self, model_type="bert"):
        self.model_type = model_type
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        if model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=3,  # positive, negative, neutral
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            self.vectorizer = TfidfVectorizer(max_features=5000)
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stopwords]
        return ' '.join(tokens)
    
    def prepare_data(self, df, text_column, label_column):
        """Prepare data for model training"""
        # Preprocess text
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df[label_column], 
            test_size=0.2, 
            random_state=42,
            stratify=df[label_column]
        )
        
        if self.model_type == "bert":
            # Tokenize data for BERT
            train_encodings = self.tokenize_data(X_train)
            test_encodings = self.tokenize_data(X_test)
            
            # Create PyTorch datasets
            train_dataset = TensorDataset(
                train_encodings['input_ids'], 
                train_encodings['attention_mask'],
                torch.tensor(y_train.values)
            )
            test_dataset = TensorDataset(
                test_encodings['input_ids'], 
                test_encodings['attention_mask'],
                torch.tensor(y_test.values)
            )
            
            return train_dataset, test_dataset
        else:
            # Vectorize data for traditional ML
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            return X_train_vec, X_test_vec, y_train, y_test
    
    def tokenize_data(self, texts):
        """Tokenize data for BERT model"""
        return self.tokenizer(
            list(texts), 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
    
    def train(self, train_data, epochs=4, batch_size=32):
        """Train the model"""
        if self.model_type == "bert":
            train_dataloader = DataLoader(
                train_data,
                sampler=RandomSampler(train_data),
                batch_size=batch_size
            )
            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            # Training loop
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}")
                self.model.train()
                total_loss = 0
                
                for batch in train_dataloader:
                    b_input_ids = batch[0].to(device)
                    b_attention_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)
                    
                    self.model.zero_grad()
                    
                    outputs = self.model(
                        b_input_ids,
                        attention_mask=b_attention_mask,
                        labels=b_labels
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    loss.backward()
                    optimizer.step()
                
                avg_train_loss = total_loss / len(train_dataloader)
                print(f"Average training loss: {avg_train_loss}")
                
        else:
            # Implement traditional ML model training here
            pass
    
    def evaluate(self, test_data):
        """Evaluate the model"""
        if self.model_type == "bert":
            test_dataloader = DataLoader(
                test_data,
                sampler=SequentialSampler(test_data),
                batch_size=32
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            self.model.eval()
            predictions = []
            true_labels = []
            
            for batch in test_dataloader:
                b_input_ids = batch[0].to(device)
                b_attention_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                
                with torch.no_grad():
                    outputs = self.model(
                        b_input_ids,
                        attention_mask=b_attention_mask
                    )
                
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(b_labels.cpu().numpy())
            
            accuracy = accuracy_score(true_labels, predictions)
            report = classification_report(true_labels, predictions)
            
            return accuracy, report
        else:
            # Implement traditional ML model evaluation here
            pass
    
    def predict_sentiment(self, texts):
        """Predict sentiment for new texts"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        if self.model_type == "bert":
            encodings = self.tokenize_data(processed_texts)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            self.model.eval()
            
            with torch.no_grad():
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            # Map predictions to sentiment labels
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            return [sentiment_map[pred] for pred in predictions]
        else:
            # Implement traditional ML prediction here
            pass


class PoliticalTrendAnalyzer:
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer
    
    def analyze_temporal_trends(self, df, text_column, timestamp_column, time_unit='day'):
        """Analyze sentiment trends over time"""
        # Preprocess text
        df['processed_text'] = df[text_column].apply(self.sentiment_analyzer.preprocess_text)
        
        # Predict sentiment
        df['sentiment'] = self.sentiment_analyzer.predict_sentiment(df['processed_text'])
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Group by time unit
        if time_unit == 'day':
            df['time_group'] = df[timestamp_column].dt.date
        elif time_unit == 'week':
            df['time_group'] = df[timestamp_column].dt.isocalendar().week
        elif time_unit == 'month':
            df['time_group'] = df[timestamp_column].dt.month
        
        # Calculate sentiment distribution over time
        sentiment_over_time = df.groupby(['time_group', 'sentiment']).size().unstack(fill_value=0)
        
        # Calculate sentiment scores
        sentiment_over_time['sentiment_score'] = (
            sentiment_over_time['positive'] - sentiment_over_time['negative']
        ) / sentiment_over_time.sum(axis=1)
        
        return sentiment_over_time
    
    def analyze_topic_sentiment(self, df, text_column, topics):
        """Analyze sentiment by topic"""
        # Preprocess text
        df['processed_text'] = df[text_column].apply(self.sentiment_analyzer.preprocess_text)
        
        # Predict sentiment
        df['sentiment'] = self.sentiment_analyzer.predict_sentiment(df['processed_text'])
        
        # Create topic flags
        for topic in topics:
            df[f'topic_{topic}'] = df[text_column].str.contains(topic, case=False)
        
        # Analyze sentiment by topic
        topic_sentiment = {}
        for topic in topics:
            topic_df = df[df[f'topic_{topic}']]
            sentiment_counts = topic_df['sentiment'].value_counts(normalize=True)
            sentiment_score = sentiment_counts.get('positive', 0) - sentiment_counts.get('negative', 0)
            topic_sentiment[topic] = {
                'sentiment_distribution': sentiment_counts,
                'sentiment_score': sentiment_score,
                'sample_size': len(topic_df)
            }
        
        return topic_sentiment
    
    def predict_approval_rating(self, sentiment_data, historical_approval=None):
        """Predict approval ratings from sentiment data"""
        if historical_approval is not None:
            # Train a model to predict approval from sentiment scores
            # This is a simplified example - would need proper training/validation in practice
            from sklearn.linear_model import LinearRegression
            
            X = pd.DataFrame({'sentiment_score': sentiment_data['sentiment_score']})
            y = historical_approval['approval_rating']
            
            model = LinearRegression()
            model.fit(X, y)
            
            predicted_approval = model.predict(X)
            return predicted_approval
        else:
            # Simple heuristic mapping sentiment to approval
            return 50 + (sentiment_data['sentiment_score'] * 30)
    
    def visualize_sentiment_trends(self, sentiment_over_time):
        """Visualize sentiment trends over time"""
        plt.figure(figsize=(12, 8))
        
        # Plot raw sentiment counts
        ax1 = plt.subplot(2, 1, 1)
        sentiment_over_time[['positive', 'neutral', 'negative']].plot(
            kind='bar', 
            stacked=True, 
            colormap='RdYlGn',
            ax=ax1
        )
        ax1.set_title('Raw Sentiment Counts Over Time')
        ax1.set_ylabel('Number of Posts')
        
        # Plot sentiment score
        ax2 = plt.subplot(2, 1, 2)
        sentiment_over_time['sentiment_score'].plot(
            kind='line', 
            marker='o',
            color='purple',
            ax=ax2
        )
        ax2.set_title('Net Sentiment Score Over Time')
        ax2.set_ylabel('Sentiment Score (-1 to 1)')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_topic_sentiment(self, topic_sentiment):
        """Visualize sentiment by topic"""
        topics = list(topic_sentiment.keys())
        scores = [topic_sentiment[topic]['sentiment_score'] for topic in topics]
        sample_sizes = [topic_sentiment[topic]['sample_size'] for topic in topics]
        
        # Normalize sample sizes for visualization
        sizes = [50 + (s / max(sample_sizes) * 200) for s in sample_sizes]
        
        plt.figure(figsize=(12, 8))
        
        # Create color map based on sentiment scores
        colors = ['red' if s < -0.1 else 'green' if s > 0.1 else 'gray' for s in scores]
        
        plt.scatter(range(len(topics)), scores, s=sizes, c=colors, alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.xticks(range(len(topics)), topics, rotation=45, ha='right')
        plt.title('Sentiment Analysis by Topic')
        plt.ylabel('Sentiment Score (-1 to 1)')
        plt.grid(axis='y', alpha=0.3)
        
        # Add sample size legend
        for i, topic in enumerate(topics):
            plt.annotate(
                f"n={sample_sizes[i]}", 
                (i, scores[i]), 
                xytext=(5, 5), 
                textcoords='offset points'
            )
        
        plt.tight_layout()
        return plt.gcf()


class ElectionPredictor:
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer
        self.regression_model = None
    
    def extract_features(self, df, text_column, candidate_keywords, issue_keywords):
        """Extract features from social media data for election prediction"""
        features = {}
        
        # Overall sentiment
        df['sentiment'] = self.sentiment_analyzer.predict_sentiment(df[text_column])
        features['overall_sentiment'] = df['sentiment'].value_counts(normalize=True).to_dict()
        
        # Candidate-specific sentiment
        for candidate, keywords in candidate_keywords.items():
            candidate_mask = df[text_column].apply(
                lambda x: any(keyword in x.lower() for keyword in keywords)
            )
            candidate_df = df[candidate_mask]
            
            if len(candidate_df) > 0:
                sentiment_counts = candidate_df['sentiment'].value_counts(normalize=True)
                features[f'{candidate}_sentiment'] = sentiment_counts.to_dict()
                features[f'{candidate}_volume'] = len(candidate_df) / len(df)
                features[f'{candidate}_net_sentiment'] = (
                    sentiment_counts.get('positive', 0) - sentiment_counts.get('negative', 0)
                )
            else:
                features[f'{candidate}_sentiment'] = {'positive': 0, 'neutral': 0, 'negative': 0}
                features[f'{candidate}_volume'] = 0
                features[f'{candidate}_net_sentiment'] = 0
        
        # Issue-specific sentiment
        for issue, keywords in issue_keywords.items():
            issue_mask = df[text_column].apply(
                lambda x: any(keyword in x.lower() for keyword in keywords)
            )
            issue_df = df[issue_mask]
            
            if len(issue_df) > 0:
                issue_sentiment = issue_df['sentiment'].value_counts(normalize=True)
                features[f'{issue}_sentiment'] = issue_sentiment.to_dict()
                features[f'{issue}_volume'] = len(issue_df) / len(df)
            else:
                features[f'{issue}_sentiment'] = {'positive': 0, 'neutral': 0, 'negative': 0}
                features[f'{issue}_volume'] = 0
        
        # Engagement metrics (if available)
        if 'retweets' in df.columns:
            features['avg_retweets'] = df['retweets'].mean()
            
            for candidate, keywords in candidate_keywords.items():
                candidate_mask = df[text_column].apply(
                    lambda x: any(keyword in x.lower() for keyword in keywords)
                )
                candidate_df = df[candidate_mask]
                
                if len(candidate_df) > 0:
                    features[f'{candidate}_avg_retweets'] = candidate_df['retweets'].mean()
                else:
                    features[f'{candidate}_avg_retweets'] = 0
        
        return features
    
    def train_prediction_model(self, feature_data, historical_results):
        """Train a model to predict election results from social media features"""
        # Convert feature dictionaries to dataframe
        X = pd.DataFrame(feature_data)
        y = pd.Series(historical_results)
        
        # Train linear regression model
        from sklearn.linear_model import LinearRegression
        self.regression_model = LinearRegression()
        self.regression_model.fit(X, y)
        
        return self.regression_model
    
    def predict_election_outcome(self, features):
        """Predict election outcome from social media features"""
        if self.regression_model is None:
            # Simple heuristic if no trained model
            candidates = [c.split('_')[0] for c in features.keys() if c.endswith('_net_sentiment')]
            predictions = {}
            
            for candidate in candidates:
                net_sentiment = features.get(f'{candidate}_net_sentiment', 0)
                volume = features.get(f'{candidate}_volume', 0)
                predictions[candidate] = 0.5 + (net_sentiment * 0.3) + (volume * 0.2)
            
            # Normalize to sum to 1
            total = sum(predictions.values())
            if total > 0:
                predictions = {k: v/total for k, v in predictions.items()}
            
            return predictions
        else:
            # Use trained model
            X = pd.DataFrame([features])
            predicted = self.regression_model.predict(X)[0]
            return predicted
    
    def visualize_predictions(self, predictions):
        """Visualize election predictions"""
        plt.figure(figsize=(10, 6))
        
        candidates = list(predictions.keys())
        scores = list(predictions.values())
        
        # Sort by prediction scores
        sorted_indices = np.argsort(scores)[::-1]
        candidates = [candidates[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Create color map
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(candidates)))
        
        plt.bar(candidates, scores, color=colors)
        plt.title('Election Outcome Prediction')
        plt.ylabel('Predicted Vote Share')
        plt.ylim(0, max(scores) * 1.2)
        
        # Add percentage labels
        for i, v in enumerate(scores):
            plt.text(i, v + 0.01, f'{v:.1%}', ha='center')
        
        plt.tight_layout()
        return plt.gcf()


# Demo usage
def demo_with_sample_data():
    # Create sample data
    np.random.seed(42)
    
    # Create dates
    dates = pd.date_range(start='2024-01-01', end='2024-04-01', freq='D')
    
    # Sample tweets about candidates
    candidates = ['Smith', 'Johnson']
    topics = ['economy', 'healthcare', 'education', 'immigration']
    
    tweet_templates = [
        "I think {} is doing a great job on {}. #election2024",
        "I don't support {}'s stance on {}. Terrible policy! #politics",
        "{} just announced a new plan for {}. What do you think?",
        "Can't believe what {} said about {} today. #news",
        "The latest poll shows {} is leading on {} issues."
    ]
    
    # Generate sample data
    data = []
    for _ in range(1000):
        date = np.random.choice(dates)
        candidate = np.random.choice(candidates)
        topic = np.random.choice(topics)
        template = np.random.choice(tweet_templates)
        tweet = template.format(candidate, topic)
        
        # Add sentiment bias based on candidate and topic
        sentiment_bias = 0
        if candidate == 'Smith':
            sentiment_bias += 0.2
        if topic == 'economy':
            sentiment_bias += 0.1
        if topic == 'immigration' and candidate == 'Johnson':
            sentiment_bias += 0.3
        
        # Determine sentiment
        rand = np.random.random()
        if rand < (0.3 + sentiment_bias):
            sentiment = 'positive'
        elif rand < (0.7 + sentiment_bias):
            sentiment = 'neutral'
        else:
            sentiment = 'negative'
        
        # Add engagement metrics
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
    
    # Initialize sentiment analyzer (mock for demo)
    class MockSentimentAnalyzer:
        def predict_sentiment(self, texts):
            return df['sentiment']
        
        def preprocess_text(self, text):
            return text
    
    sentiment_analyzer = MockSentimentAnalyzer()
    
    # Analyze political trends
    trend_analyzer = PoliticalTrendAnalyzer(sentiment_analyzer)
    sentiment_over_time = trend_analyzer.analyze_temporal_trends(df, 'text', 'date', 'week')
    topic_sentiment = trend_analyzer.analyze_topic_sentiment(df, 'text', topics)
    
    # Predict election outcome
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
    
    # Create visualizations
    trend_viz = trend_analyzer.visualize_sentiment_trends(sentiment_over_time)
    topic_viz = trend_analyzer.visualize_topic_sentiment(topic_sentiment)
    election_viz = election_predictor.visualize_predictions(predictions)
    
    # Return results
    return {
        'sentiment_over_time': sentiment_over_time,
        'topic_sentiment': topic_sentiment,
        'election_predictions': predictions,
        'visualizations': {
            'trend_viz': trend_viz,
            'topic_viz': topic_viz,
            'election_viz': election_viz
        }
    }

# Run demo if executed as script
if __name__ == "__main__":
    results = demo_with_sample_data()
    print("Sentiment over time:", results['sentiment_over_time'])
    print("Topic sentiment:", results['topic_sentiment'])
    print("Election predictions:", results['election_predictions'])
    plt.show()