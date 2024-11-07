import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')


text = "I'm excited about learning NLP! It's amazing how much we can do with text processing."


tokens = word_tokenize(text)


stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]


sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores(text)


print("Original Tokens:", tokens)
print("Filtered Tokens:", filtered_tokens)
print("Number of Tokens:", len(tokens))
print("Number of Tokens after Stop Word Removal:", len(filtered_tokens))
print("Sentiment Scores:", sentiment)


if sentiment['compound'] >= 0.05:
    print("Overall Sentiment: Positive")
elif sentiment['compound'] <= -0.05:
    print("Overall Sentiment: Negative")
else:
    print("Overall Sentiment: Neutral")
