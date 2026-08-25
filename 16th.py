from collections import Counter
import re

# Read the text file
with open("sample_text.txt", "r") as file:
    text = file.read()

# Convert text to lowercase
text = text.lower()

# Remove punctuation and extract words
words = re.findall(r'\b\w+\b', text)

# Calculate word frequency
frequency = Counter(words)

# Display frequency distribution
print("Word Frequency Distribution:")

for word, count in frequency.items():
    print(word, ":", count)