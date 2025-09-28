import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#PERFORM DATA CLEANING,AGGREGATION AND FILTERING
#load the datasets
df = pd.read_csv(r"C:\Users\ARYA\My Learning\EDA\titanic.csv")

#search null values in a specific column
print(df.isnull().sum()[df.isnull().sum() > 0])

#inspect data  
print(df.info())
print(df.describe())

#handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if x == "Unknown" else 1)

#remove duplicate
df = df.drop_duplicates()

#filter data : passanger in first class
first_class = df[df["Pclass"]==1]
print("first class passengers: \n", first_class.head())

#GENERATE VISUALIZATIONS TO ILLUSTRATE KEY INSIGHTS
#Bar Chart : Survival rate by class
survival_by_class = df.groupby("Pclass")["Survived"].mean()
survival_by_class.plot(kind = "bar", color="skyblue")
plt.title("survival rate by class")
plt.ylabel("Survival rate")
plt.show()

#Survival by gender
survival_by_sex = df.groupby("Sex")["Survived"].mean()
survival_by_sex.plot(kind='bar', color=['lightcoral', 'skyblue']) 
plt.title("Survival Rate by Gender")
plt.ylabel("Average Survival Rate")
plt.xlabel("Gender")
plt.show()

#Histogram : Age distribution
sns.histplot(df["Age"], kde=True, bins=20, color="brown")
plt.title("Age Distribution")
plt.ylabel("Frequency")
plt.xlabel("Age")
plt.show()

#Scatter plot : Age vs Fare
plt.scatter(df["Age"], df["Fare"], alpha= 0.5, color = "green")
plt.ylabel("Fare")
plt.xlabel("Age")
plt.title("Age vs Fare")
plt.show()