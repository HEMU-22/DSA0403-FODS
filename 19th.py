import pandas as pd
import re
from collections import Counter

# Create customer reviews data
data = pd.DataFrame({
    "Review": [
        "Good product and good quality",
        "Excellent product and fast delivery",
        "Good quality and excellent service"
    ]
})

# Store all words
all_words = []

# Process each review
for review in data["Review"]:
    # Convert to lowercase
    review = review.lower()

    # Remove punctuation and extract words
    words = re.findall(r'\b\w+\b', review)

    # Add words to list
    all_words.extend(words)

# Calculate frequency distribution
frequency = Counter(all_words)

# Display result
print("Word Frequency Distribution:")

for word, count in frequency.most_common():
    print(word, ":", count)