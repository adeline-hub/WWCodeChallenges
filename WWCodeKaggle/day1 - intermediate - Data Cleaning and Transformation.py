import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from google.colab import drive
drive.mount('/content/drive')

# 1. Data Cleaning and Transformation

https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023


##Handle missing values and perform initial data transformations.

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/WWCode Data Science/datasets/world-data-2023.csv')

# Display the first few rows of the DataFrame
df.head()

##a) Handle missing values

###1. Removing Rows with Missing Values:

# Display rows with missing values
rows_with_missing_values = df[df.isnull().any(axis=1)]
#rows_with_missing_values

# Display total count of missing values for each column
missing_values_count = df.isnull().sum()
missing_values_count


# Drop rows with any missing values
df_cleaned = df.dropna()

df_cleaned.head()

###2. Filling Missing Values::

# Fill missing values with a specific value (e.g., mean, median)
df_filled = df.fillna(df.mean())

df_filled.head()

###3. Interpolation:

interpolate() function is basically used to fill NA values in the dataframe or series. But, this is a very powerful function to fill the missing values. It uses various interpolation technique to fill the missing values rather than hard-coding the value.

# Interpolate missing values
df_interpolated = df.interpolate()

df_interpolated



##b) Initial Data Transformations

###1. Changing Data Types:

# Convert a column to a different data type (e.g., from object to datetime)
#df['Date'] = pd.to_datetime(df['Date'])

###2. Adding New Columns:

# Create a new column based on existing columns
#df['Total'] = df['Quantity'] * df['Price']

###3. Dropping Columns:

# Drop unnecessary columns
#df_dropped = df.drop(['Column1', 'Column2'], axis=1)

###4. Renaming Columns:

# Rename columns for clarity
#df.rename(columns={'OldName': 'NewName'}, inplace=True)

###5. Handling Duplicates:

# Remove duplicate rows
df_no_duplicates = df.drop_duplicates()

###6. Convert object columns to numeric data types:


df.info()

# Remove '%' from the 'Percentage' column
df['Agricultural Land( %)']= df['Agricultural Land( %)'].str.replace('%', '')

# Convert the column to numeric (optional)
df['Agricultural Land( %)'] = pd.to_numeric(df['Agricultural Land( %)'])

# List of percentage columns to process
columns_to_process = ['CPI Change (%)', 'Forested Area (%)', 
                      'Gross primary education enrollment (%)', 'Gross tertiary education enrollment (%)', 
                      'Population: Labor force participation (%)', 'Tax revenue (%)','Total tax rate','Unemployment rate','Fertility Rate','Birth Rate','Out of pocket health expenditure']


columns_to_numerical =  ['Land Area(Km2)', 'Armed Forces size', 
                      'Co2-Emissions', 'Gasoline Price','GDP',
                      'Population', 'Urban_population']

# Remove commas, $, % and convert selected columns to numeric
df[columns_to_numerical] = df[columns_to_numerical].apply(lambda x: x.astype(str).str.replace(',', '').str.replace('%', '').str.replace('$', '').str.strip().astype(float))

df.head()

# Convert all object columns to numeric data types
#df = df.apply(pd.to_numeric, errors='coerce')

###7. Binning Numerical Data:

# Bin numerical data into categories
bins = [0, 50, 75, 100]
labels = ['Low', 'Medium', 'High']
df['Category'] = pd.cut(df['Agricultural Land( %)'], bins=bins, labels=labels)

df.head()
