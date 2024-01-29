import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from google.colab import drive
drive.mount('/content/drive')

from openpyxl.styles import NamedStyle

# 0. Cleaning and transformation


##Identify potential predictive variables.

https://www.kaggle.com/datasets/rithykka/r-and-d-grant-data-tracker-analysis/

#df = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/WWCode Data Science/datasets/Grant Data Export.xlsx')

# Assuming 'Grant Data Export.xlsx' is in the input folder
file_path = '/content/drive/MyDrive/Colab Notebooks/WWCode Data Science/datasets/Grant Data Export.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows of the DataFrame
df.head()

df.info()

.................................................................................

##a) Cleaning

# Display total count of missing values for each column
missing_values_count = df.isnull().sum()
missing_values_count

# Remove 'FY' from the 'Percentage' column
df['Year']= df['Year'].str.replace('FY', '')

##b) Transformation

# Assuming 'Year' is the column you want to convert
df['Year'] = pd.to_datetime(df['Year'], errors='coerce')

# Extract the year and create a new column 'Year'
df['Year'] = df['Year'].dt.year

# Drop the original 'Date_Column' if needed
# df.drop('Year', axis=1, inplace=True)

df.head()

.............................................................................................



# 1. Predictive Insights from exploratory data analysis (EDA)


##Identify potential predictive variables.

https://www.kaggle.com/datasets/rithykka/r-and-d-grant-data-tracker-analysis/


In exploratory data analysis (EDA), the goal is to understand the main characteristics of your dataset and identify patterns or relationships between variables. While EDA itself doesn't provide predictive insights, it helps you identify potential predictive variables that can be further explored using statistical modeling or machine learning techniques.



##a) Correlation Analysis

# Correlation matrix
correlation_matrix = df.corr()

# Visualize correlations using a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

##b) Pairwise Plots

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you want to include all numerical columns in the pairplot
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Create pairwise scatter plots
sns.pairplot(df[numerical_columns])
plt.show()



##c) Distribution Analysis

# Distribution of the target variable
sns.histplot(df['Year'], kde=True)
plt.title("Distribution of Target Variable")
plt.show()

# Distribution of predictor variable
sns.histplot(df['Amount(USD)'], kde=True)
plt.title("Distribution of Predictor Variable")
plt.show()

##d) Outlier Detection

# Boxplot for outlier detection
sns.boxplot(x=df['Amount(USD)'])
plt.title("Boxplot for Outlier Detection")
plt.show()

##e) Categorical Variable Analysis

# Count plot for categorical variable
sns.countplot(x='Funder Country', data=df)
plt.title("Count Plot for Categorical Variable")
plt.show()

##f) Feature Importance:

from sklearn.ensemble import RandomForestClassifier

# Assume 'target_variable' is a binary classification target
X = df.drop('target_variable', axis=1)
y = df['target_variable']

# Train a RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Feature importance plot
feature_importance = pd.Series(clf.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title("Feature Importance")
plt.show()


##g) Domain Knowledge



##h) Dimensionality Reduction

from sklearn.decomposition import PCA

# Assume X contains the predictor variables
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Scatter plot of reduced dimensions
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
plt.title("Scatter Plot of Reduced Dimensions")
plt.show()


##s1) Frequency Analysis


Analyze the frequency of different values in each string column.

Identify the most common and least common values.

Look for patterns or anomalies in the distribution of string values.

# Analyze frequency of values in a string column
frequency_counts = df['Disease/Health area'].value_counts()

# Plot horizontal bar plot
plt.figure(figsize=(10, 6))
frequency_counts.sort_values().plot(kind='barh', color='skyblue')
plt.xlabel('Number of Unique Values')
plt.ylabel('Columns')
plt.title('Horizontal Bar Plot of Unique Values per Column')
plt.show()

# Assuming 'df' is your DataFrame
unique_values_per_column = df.nunique()

# Display the count of unique values per column
unique_values_per_column

import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame and 'Funding Type' is the column you want to plot
specific_column = 'Funding Type'
value_counts = df[specific_column].value_counts()

# Plot pie chart
plt.figure(figsize=(10, 6))
plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title(f'Distribution of {specific_column}', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.9), fontsize=10)
plt.show()

##s2) Text Length Analysis


Explore the length of strings in each column.

Create a new feature representing the length of the strings.

Investigate if there's a relationship between string length and the target variable.

# Create a new feature representing the length of strings
#df['String_Length'] = df['String_Column'].apply(len)
#print(df[['String_Column', 'String_Length']])

##s3) Text Preprocessing


Perform text preprocessing steps such as lowercasing, removing stop words, and stemming/lemmatization.

Create new features based on preprocessed text data.

Explore the relationships between the preprocessed text features and the target variable.

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Example text preprocessing functions
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = [ps.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(text)

# Apply text preprocessing to a string column
df['Processed_Text'] = df['Disease/Health area'].apply(preprocess_text)
print(df[['Disease/Health area', 'Processed_Text']])

##s4)  N-grams Analysis


Analyze n-grams (sequences of words) in the text data.

Create features based on n-grams and explore their relationships with the target variable.

Consider using techniques like TF-IDF (Term Frequency-Inverse Document Frequency) to represent the importance of words.

from sklearn.feature_extraction.text import CountVectorizer

# Analyze n-grams in a string column
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))
ngram_matrix = ngram_vectorizer.fit_transform(df['String_Column'])
ngram_features = ngram_vectorizer.get_feature_names_out()
df_ngrams = pd.DataFrame(ngram_matrix.toarray(), columns=ngram_features)
print(df_ngrams)

##s5) Word Clouds

Create word clouds for each string column to visually identify the most common words.

Observe if certain words stand out in relation to the target variable.

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create a word cloud for a string column
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Disease/Health area']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

##s6) Topic Modeling


Apply topic modeling techniques (e.g., Latent Dirichlet Allocation - LDA) to identify topics in the text data.

Explore the relationship between topics and the target variable.

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Apply LDA for topic modeling
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Disease/Health area'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)
topics = lda.transform(X)
print(topics)

##s7) Feature Engineering


Create new features based on domain knowledge or specific patterns in the text data.

Explore the relationship between these engineered features and the target variable.

# Create new features based on domain knowledge
df['Contains_Keyword'] = df['String_Column'].str.contains('keyword').astype(int)
print(df[['String_Column', 'Contains_Keyword']])


##s8)  Advanced Natural Language Processing (NLP) Techniques


If your dataset involves natural language, consider using more advanced NLP techniques like word embeddings (e.g., Word2Vec, GloVe) or transformer-based models (e.g., BERT).

Utilize pre-trained language models for feature extraction.

# Example using Word2Vec from gensim
from gensim.models import Word2Vec

# Train Word2Vec model on a tokenized text column
tokenized_text = df['String_Column'].apply(lambda x: x.split())
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)
word_vectors = word2vec_model.wv
print(word_vectors['example_word'])
