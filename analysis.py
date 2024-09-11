import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/hyewon/Desktop/research/diabetes.csv'
df = pd.read_csv(file_path)



# df.hist(figsize=(10, 8))


# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.show()

outcome_counts = df['Outcome'].value_counts()
print(outcome_counts)