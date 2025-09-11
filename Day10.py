# --- 1. Import Libraries ---
import wikipediaapi
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re

# --- 2. Perform One-Time Downloads for NLTK (Corrected Version) ---
print("--- Checking for NLTK data ---")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading 'stopwords' corpus...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading 'wordnet' corpus...")
    nltk.download('wordnet')

print("All necessary NLTK data is available.")


# --- 3. Fetch Text from Wikipedia ---
print("\n--- Fetching Text from Wikipedia ---")
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='My30DaysOfAIProject/1.0'
)
page_ai = wiki_wiki.page('Artificial intelligence')

if page_ai.exists():
    text_corpus = page_ai.text
    print("Successfully fetched the Wikipedia article on 'Artificial intelligence'.")
else:
    print("Wikipedia page not found. Using placeholder text.")
    text_corpus = "artificial intelligence machine learning data science model algorithm"


# --- 4. Preprocess the Text ---
print("\n--- Preprocessing the Text ---")

# Convert text to lowercase and tokenize
tokens = word_tokenize(text_corpus.lower())

# Remove punctuation and non-alphabetic characters
words = [word for word in tokens if word.isalpha()]

# Remove stop words (common words like 'the', 'a', 'is')
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]

# Lemmatize the words (reduce them to their root form)
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

# Join the words back into a single string
processed_text = " ".join(lemmatized_words)
print("Text preprocessing complete.")


# --- 5. Load the Custom Mask Image ---
print("\n--- Loading Custom Mask ---")
try:
    # IMPORTANT: Make sure you have an image named 'mask.png' in the same folder.
    # A simple silhouette (e.g., a cloud, brain, or lightbulb) works best.
    # The words will fill the WHITE area of the image.
    custom_mask = np.array(Image.open("mask.png"))
    print("Custom mask image 'mask.png' loaded successfully.")
except FileNotFoundError:
    print("Mask image 'mask.png' not found. A default rectangular shape will be used.")
    custom_mask = None


# --- 6. Generate and Display the Word Cloud ---
print("\n--- Generating Word Cloud ---")

# Initialize the WordCloud object with settings
wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color='white',
    mask=custom_mask,
    contour_width=3,
    contour_color='steelblue',
    colormap='viridis' # A nice color palette for the words
)

# Generate the word cloud from the processed text
wordcloud.generate(processed_text)

# Display the generated image using matplotlib
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off") # Hide the axes
plt.title("Word Cloud for 'Artificial Intelligence'", fontsize=20, pad=20)
plt.show()

print("\nWord cloud generation complete!")