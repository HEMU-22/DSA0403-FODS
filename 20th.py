import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt

# Read CSV file
data = pd.read_csv("data.csv")

# Stop words
stop_words = {
    "the", "and", "is", "a", "an", "of",
    "to", "in", "for", "on", "with", "this",
    "that", "it", "was", "are", "very"
}

# Store all words
all_words = []

# Preprocess feedback
for feedback in data["feedback"].dropna():

    # Convert to lowercase
    feedback = feedback.lower()

    # Remove punctuation
    feedback = re.sub(r'[^\w\s]', '', feedback)

    # Split into words
    words = feedback.split()

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    # Add words to list
    all_words.extend(words)

# Calculate frequency distribution
frequency = Counter(all_words)

# Get user input
N = int(input("Enter number of top words: "))

# Get top N words
top_words = frequency.most_common(N)

print("\nTop", N, "Most Frequent Words:")

for word, count in top_words:
    print(word, ":", count)

# Prepare data for graph
words = [item[0] for item in top_words]
counts = [item[1] for item in top_words]

# Create bar graph
plt.bar(words, counts)
plt.title("Top Frequent Words in Customer Feedback")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
