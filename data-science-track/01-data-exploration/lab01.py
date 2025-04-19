import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('students.csv')
df.head()   # Top 5 rows
df.info()   # Data types & null counts
df.describe()   # Summary stats for numbers

df.isnull().sum() # Null counts

print("\n\nColumns:", df.columns.tolist())
print("Unique Genders:", df['gender'].unique())
print("Score range:", df['gpa'].min(), "-", df['gpa'].max())
print("\n\n")

# Visualise missing data
# sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
# plt.title("Missing Data Heatmap")
# plt.show()

# Data Cleaning Summary:
# - Age: Left as NaN (optional demographic data)
# - Credits Completed: Left as NaN (could indicate incomplete records or new students)
# - Attendance Rate: Filled NaN with 0% (assumed missing means no recorded attendance)
# - GPA: Rounded to 2 decimal places
# - Dataset sorted by student_id for consistency
df['attendance_rate'] = df['attendance_rate'].fillna(0)
df['gpa'] = df['gpa'].round(2)
df = df.sort_values(by='student_id')

# Export cleaned data
df.to_csv('students-clean.csv', index=False)
df_clean = pd.read_csv('students-clean.csv')

# GPA distribution and frequency
sns.histplot(df_clean['gpa'], kde=True)
plt.title("GPA Distribution")
plt.xlabel("GPA")
plt.ylabel("Frequency")
plt.savefig('gpa-distribution.png')
plt.close()

# Does attenance affect GPA?
sns.scatterplot(x='attendance_rate', y='gpa', data=df)
plt.title("Attendance Rate vs GPA")
plt.xlabel("Attendance Rate (%)")
plt.ylabel("GPA")
plt.savefig('attendance-vs-gpa.png')
plt.close()
