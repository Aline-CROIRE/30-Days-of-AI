import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

# Create a list of sample movie reviews
reviews = [
    "This movie was absolutely fantastic! The acting was superb.",
    "I was really disappointed. The plot was boring and predictable.",
    "The film was okay, not great but not terrible either.",
    "I have never seen a movie so wonderful and moving. A must-see!",
    "A complete waste of my time and money.",
    "The cinematography was beautiful, but the story lacked depth.",
    "It's an average movie, you can watch it if you have nothing else to do.",
    "The ending was a huge letdown.",
    "AMAZING! I loved every second of it. Truly a masterpiece!!!",
    "The script felt rushed and the characters were one-dimensional."
]

# Create a DataFrame from our list
df = pd.DataFrame(reviews, columns=['review_text'])

print("Sample dataset created successfully:")
print(df)

print("\n--- Analyzing Sentiment ---")

# 1. Initialize the VADER Sentiment Intensity Analyzer
analyzer = SentimentIntensityAnalyzer()


test_sentence = "The movie was not great."
scores = analyzer.polarity_scores(test_sentence)
print(f"Scores for '{test_sentence}': {scores}")


df['sentiment_scores'] = df['review_text'].apply(lambda review: analyzer.polarity_scores(review))

print("\nSentiment scores added to the DataFrame:")
print(df)
print("\n--- Categorizing Sentiment ---")

# 1. Create a function to categorize based on the compound score
def get_sentiment_category(scores):
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# 2. Apply this function to our 'sentiment_scores' column
df['sentiment_category'] = df['sentiment_scores'].apply(get_sentiment_category)

# Let's also extract the compound score for easier viewing
df['compound_score'] = df['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])


print("\nFinal DataFrame with sentiment categories:")
print(df[['review_text', 'compound_score', 'sentiment_category']])

print("\n--- Visualizing Sentiment Distribution ---")

# 1. Count the number of reviews in each category
sentiment_counts = df['sentiment_category'].value_counts()

# 2. Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%', # This formats the percentages on the chart
        startangle=140,
        colors=['#66c2a5', '#fc8d62', '#8da0cb']) # A nice color scheme

plt.title('Sentiment Distribution of Movie Reviews', fontsize=16)
plt.ylabel('') # Hides the 'sentiment_category' label on the side
plt.show()

print("\nSentiment Counts:")
print(sentiment_counts)