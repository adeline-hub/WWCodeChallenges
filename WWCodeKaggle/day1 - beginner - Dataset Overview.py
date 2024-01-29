import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from google.colab import drive
drive.mount('/content/drive')


# 1. Dataset Overview
##Explore the dataset's basic structure (columns, rows, types).

file_path = '/content/drive/MyDrive/Colab Notebooks/WWCode Data Science/datasets/Customer-Churn-Records.csv'

# Read the content of the document
with open(file_path, 'r') as file:
    document_content = file.read()
#print(document_content)

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/WWCode Data Science/datasets/Customer-Churn-Records.csv')

# Display the first few rows of the DataFrame
df.head(10)

# Display basic information about the DataFrame
df.info()

# Get summary statistics for numeric columns
df.describe()

# Display the column names
df.columns

# Display the data types of each column
df.dtypes
