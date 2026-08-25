import pandas as pd
import matplotlib.pyplot as plt

# Student study time and exam scores
study_time = [1, 2, 3, 4, 5, 6, 7, 8]
scores = [45, 50, 55, 60, 65, 72, 80, 88]

# Create DataFrame
data = pd.DataFrame({
    "Study_Time": study_time,
    "Exam_Score": scores
})

# Calculate correlation
correlation = data["Study_Time"].corr(data["Exam_Score"])

print("Correlation coefficient:", correlation)

# Scatter plot
plt.scatter(data["Study_Time"], data["Exam_Score"])
plt.title("Study Time vs Exam Score")
plt.xlabel("Study Time (Hours)")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

# Line plot
plt.plot(data["Study_Time"], data["Exam_Score"], marker="o")
plt.title("Study Time vs Exam Score")
plt.xlabel("Study Time (Hours)")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

# Interpretation
if correlation > 0:
    print("There is a positive correlation between study time and exam scores.")
elif correlation < 0:
    print("There is a negative correlation between study time and exam scores.")
else:
    print("There is no correlation between study time and exam scores.")