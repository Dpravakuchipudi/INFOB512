
![Logo](https://news.luddy.indiana.edu/images/news/2021/luddygenericsocialwide1.jpg)


# INFO B 512 SCIENTIFIC AND CLINICAL DATA MANAGEMENT
# Assessing Hospital Care Quality in the Midwest: A Comparative Analysis of Performance Measures


Measuring healthcare quality has become a pressing issue that has attracted the attention of various stakeholders, including researchers, policymakers, and the public.  Improving healthcare quality is thus a top priority, but its success depends on the quality measures used to assess healthcare practices. In this study, we conducted a comprehensive evaluation of current healthcare quality measurement methods and their impact on the healthcare system in the Midwest US states. We identified significant gaps in the field, particularly in the wide variations of hospital performance. To address this, we propose improving communication among hospitals to enhance patient experience and quality of care. Quality standards are critical benchmarks that allow healthcare organizations to measure resource utilization and support improvement efforts based on objective and factual information. Evaluating the quality of hospital care and treatment methods is therefore essential to assist physicians and patients in making informed decisions. Our study aims to contribute to the development of better methods for measuring healthcare quality, which can help identify the strengths and weaknesses of the healthcare system in the Midwest US states and provide guidance for future improvements.


## Demo

Here is our Website demo link 
https://youtu.be/nH33UH7sJcM

## Project  Members

- Supraja Pericherla
- Durga pravallika Kuchipudi
- Emaaz Assadi
- Rutwik Gudipati
- Sumayya Gurmen
- Hymavathi Thota




## Project Methodology

- Data Preprocessing and Cleaning

- Designing Databases

- Developing machine learmning model and advanced statistical analysis in Python

- Generating Heatmaps and advanced visualisations in Tableau


## Technologies

**Softwares Used:** Microsoft excel, My SQL Workbench, Tableau

**Programming Languages:** 
- Python for Data Analysis 
- SQL for Database storing and Processing


## Installation

Importing Packages and loading the Dataset
```bash
  import numpy as np 
  import pandas as pd
  import matplotlib.pyplot as plt 
  import math as math
  from sklearn.linear_model import LinearRegression
  df = pd.read_csv("Quality measures (1).csv")

```
Importing Packages and loading the Dataset
```bash
df.head()
df.info()
df.drop(['QID','Provider ID'], axis=1, inplace = True) df.head()
```
Check for Null Values
```bash
df.isnull().sum()
# checking null values % in the data df_null = df.isna().mean().round(4) * 100
df_null.sort_values(ascending=False).head()
```
Replacing Null values with Appropriate Values
```bash
df['Score'] = df['Score'].replace(['Not Available'], ['0'])
df['Score'] = df['Score'].replace([''], ['-1'])
df['Score'] = df['Score'].replace(['High (40,000 - 59,999 patients annually)'], ['50000'])
df['Score'] = df['Score'].replace(['Very High (60,000+ patients annually)'], ['70000'])
df['Score'] = df['Score'].replace(['Medium (20,000 - 39,999 patients annually)'], ['30000'])
df['Score'] = df['Score'].replace(['Low (0 - 19,999 patients annually)'], ['10000'])
df['Sample'] = df['Sample'].replace(['Not Available'], ['0'])
df['Sample'] = df['Sample'].replace([''], ['0'])
cat_cols = ['Condition', 'Measure ID', 'Score', 'Sample','Footnote'] for x in cat_cols:
df[x] = df[x].fillna('0')
num_cols = ['ZIP Code','Re-Admission Rate','Emergency department wait time','Efficiency','Morality Rate']
for x in num_cols:
df[x] = df[x].fillna(0)
df.columns
print(df['ZIP Code'].unique())
print(df['Score'].unique())
print(df['Sample'].unique())
print(df['Re-Admission Rate'].unique())
```
Splitting data into X and Y and train data and test data before encoding the categorical variables because else data leakage happens
```bash
ind_columns = ['ZIP Code', 'Condition', 'Measure ID', 'Score',
	'Sample', 'Footnote', 'Re-Admission Rate',
	'Emergency department wait time', 'Efficiency']
X = df[ind_columns] Y = df['Morality Rate']
```

Splitting data in test data and train data
```bash
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
X_train.columns
X_train['ZIP Code'].unique()
```

Encoding categorical columns in train and test data using label encoder
```bash
from sklearn import preprocessing
# label_encoder object knows how to understand word labels. label_encoder = preprocessing.LabelEncoder()
# Encode labels label_encoder.fit(X_train['Condition'])
X_train['Condition']= label_encoder.transform(X_train['Condition']) 
X_test['Condition']= label_encoder.transform(X_test['Condition']) 
print(X_train['Condition'].unique())
label_encoder.fit(X_train['Measure ID'])
X_train['Measure ID']= label_encoder.transform(X_train['Measure ID']) 
X_test['Measure ID']= label_encoder.transform(X_test['Measure ID']) 
print(X_train['Measure ID'].unique())
label_encoder.fit(X_train['Footnote'])
X_train['Footnote']= label_encoder.transform(X_train['Footnote']) 
X_test['Footnote']= label_encoder.transform(X_test['Footnote']) 
print(X_train['Footnote'].unique())

```

Plotting Correlation Matrix for all Independent variables
```bash
import seaborn as sn 
corr_matrix = X_train.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()
```

Regression Model 
```bash
regressor = LinearRegression() 
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))
```
Plotting the Linear Regression Model 
```bash
y_train_arr = y_train.to_numpy()
y_train_new = y_train_arr.reshape(y_train.shape[0], 1) y_train_new.shape
import seaborn as sns
y_test
import numpy as np np.array(y_test)
y_pred
sns.scatterplot(y_test)
sns.scatterplot(y_test)
sns.scatterplot(y_test, marker="+")
sns.scatterplot(y_pred, marker="+")
sns.scatterplot(y_test, alpha=0.5)
sns.scatterplot(y_pred, alpha=0.5)
```
seaborn data visualisation
```bash
import seaborn as sns
%matplotlib inline
import pandas as pd
quality = pd.read_csv('/content/sample_data/Quality measures.csv')
sns.displot(data=quality, x="Re-Admission Rate", kde=True, bins= 30)
import seaborn as sns
sns.histplot(data=quality, x="Re-Admission Rate") sns.jointplot(x="Score", y="Re-Admission Rate", data=quality, kind='reg')
sns.jointplot(x="Score", y="Re-Admission Rate", data=quality, kind='kde')
sns.pairplot(quality,hue= 'Re-Admission Rate')
```
displot
```bash
sns.displot(data=quality, x="Re-Admission Rate", kde=True, bins= 30)
```
displot
```bash
sns.displot(data=quality, x="Re-Admission Rate", kde=True, bins= 30)
```
jointplot
```bash
sns.jointplot(x="Score", y="Re-Admission Rate", data=quality, kind='reg')
```
pairplot
```bash
sns.pairplot(quality,hue= 'Re-Admission Rate')
```


## Results


We further developed a Machine learning model and found the analysis that A negative correlation between patient experience scores and readmission rates means that hospitals with higher patient experience scores tend to have lower readmission rates, and vice versa. In other words, there is an inverse relationship between these two variables - as one variable increases, the other tends to decrease."



