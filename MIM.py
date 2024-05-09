'''
CRISP - ML(Q)

**CRISP-ML(Q) process model describes six phases:**

- Business and Data Understanding
- Data Preparation (Data Engineering)
- Model Building (Machine Learning)
- Model Evaluation and Tunning
- Deployment
- Monitoring and Maintenance

Problem Statements: Bounce rate is increasing significantly leading to patient dissatisfaction.



Business Objective: Predict the profitability of startup companies.

Business Constraints: Limited availability of comprehensive historical data, 
                      potential external factors influencing startup performance.

## Success Criteria : - 

Business Success Criteria: Increase predictive accuracy to improve decision-making regarding investment in startup companies, 
                           aiming for a 15% increase in profit.

Machine Learning Success Criteria: Achieve an R-squared value of at least 0.8 and a mean squared error (MSE) below 0.1.

Economic Success Criteria: Venture capitalists experience a 20% increase in ROI by leveraging 
                           insights from the predictive model to guide investment decisions in startup companies.

# Data Collection :

# Data Description :
    The dataset consists of 14218 entries with the following columns:

VARIABLE NAME - DESCRIPTION

Typeofsales:	Type of sale of the drug. Either the drug is sold or returned.

Patient_ID: 	ID of a patient

Specialisation:	Name of Specialisation (eg. Cardiology)

Dept:	        Pharmacy, the formulation is related with.

Dateofbill:  	Date of purchase of medicine

Quantity:	    Quantity of the drug

ReturnQuantity:	Quantity of drug returned by patient to the pharmacy

Final_Cost:	    Final Cost of the drug (Quantity included)

Final_Sales:	Final sales of drug

RtnMRP:	        MRP of returned drug (Quantity included)

Formulation:	Type of formulation

DrugName:	    Generic name of the drug

SubCat:	        Subcategory (Type) to the category of drugs

SubCat1:     	Subcategory (condition) to the category of drugs
'''

# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from feature_engine.outliers import Winsorizer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_excel("/Users/pavankumar/Projects/Medical Inventory Management/Datasets/Medical Inventory Optimaization Dataset.xlsx")

data.info()

# Connection to databses
user = 'root'
pw = '967699'
db = 'datasets'
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

data.to_sql("Medical Inventory", con = engine, if_exists = "replace", chunksize = 1000, index = False)

# Load the data from My sql
sql = 'select * from `datasets`.`Medical Inventory`'
data = pd.read_sql_query(sql, con = engine)


#*************** Data Preparation ***********************


# Type casting 
data["Patient_ID"] = data["Patient_ID"].astype('str')
data["Final_Sales"] = data["Final_Sales"].astype('float32')
data["Final_Cost"] = data["Final_Cost"].astype('float32')

# Checking for any null values 
data.isna().sum() 

# We see there are few missimg class values
# Handling Missing values
for col in ['Formulation','DrugName','SubCat','SubCat1']:
    print(data[col].unique(),"\n")
    print(data[col].value_counts().unique().size,'\n')

data.reset_index(drop = True, inplace = True)

# Imputing missing values based on mode of groups   
group_cols = ['Typeofsales', 'Specialisation', 'Dept']

for col in ['Formulation', 'DrugName', 'SubCat', 'SubCat1']:
    # Reset the index before applying group-wise mode
    data[col] = data.groupby(group_cols)[col].transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
data.isna().sum() 

# we still see few missing values
data.dropna(inplace = True)
data.reset_index(drop = True, inplace = True)
data.isna().sum() 

# Handling Duplicates
duplicates = data.duplicated()
sum(duplicates)

# Removing duplicated
data = data.drop_duplicates()
duplicates = data.duplicated()
sum(duplicates)

## Data Manupulation
   
# Assuming 'Date' is the column containing dates in your DataFrame
data['Dateofbill'] = pd.to_datetime(data['Dateofbill'], format='mixed', dayfirst= True)

data = data.sort_values(by = 'Dateofbill', ascending  = True )

# Speifying columns Final cost and final sale  to round 


# round the values in the column to zero
data['Final_Cost'] = data['Final_Cost'].apply(lambda x: round(x))
data['Final_Sales'] = data['Final_Sales'].apply(lambda x: round(x))


data.head(10)

data.describe()


# Drop Irrelavent columns
data.drop(['Patient_ID','ReturnQuantity'], axis = True, inplace = True)

# Segregate numeric and categorical cloumns
numeric_features = data.select_dtypes(exclude = ['object','datetime64']).columns
numeric_features

categorical_features = data.select_dtypes(include =['object']).columns
categorical_features


# FIRST MOMENT BUSSINESS DECESION / Measure of Central Tendency

# Mean
data[numeric_features].mean()

# Median
data[numeric_features].median()

# Mode
data.mode()

# Second Moment Decesion / Measure of Dispersion

# Varince
data[numeric_features].var()

# Standard Deviation
data[numeric_features].std()

# 3rd Moment Decesion
# Skewness
data[numeric_features].skew()

# 4 Moment Decesion 
#Kurtosis
data[numeric_features].kurt()

#EDA 
data.Quantity.max()
plt.hist(data.Quantity,color = 'orange', bins = 20, alpha =1)
plt.xlim(0, 160)

data.Final_Cost.max()
plt.hist(data.Final_Cost,color = 'orange', bins = 500, alpha = 1)
plt.xlim(0, 3500)


data.Final_Sales.max()
plt.hist(data.Final_Sales,color = 'orange', bins = 500, alpha = 1)
plt.xlim(0, 4000)


data.RtnMRP.max()
plt.hist(data.RtnMRP,color = 'orange', bins = 500, alpha = 1)
plt.xlim(0, 1000)


# Convert date Format to Month
data['Dateofbill'] = data['Dateofbill'].dt.strftime("%b")

data.head()

# Pivot the DataFeame based on SubCat of Drugs
data_pivoted = data.pivot_table(index = 'SubCat', columns = 'Dateofbill', values = "Quantity")






# Segregate Input variables with Target variable
X = data.iloc[:,1:]
Y = data.iloc[:,0]

# Auto EDA
# =============================================================================
# import sweetviz
# sv = sweetviz.analyze(data)
# sv.show_html('The Report.html')
# 
# import dtale
# d = dtale.show(data)
# d.open_browser()
# =============================================================================

# Here we observe that in the type_of_sales (88%) have been saled and only 12% have been returned back by the customers

# Segregate numeric and categorical cloumns
numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features

categorical_features = X.select_dtypes(include =['object']).columns
categorical_features

# Outlier analysis
data[numeric_features].plot(kind = 'box', subplots = True, sharey = False, figsize = (14,8), fontsize = 18)
plt.subplots_adjust(wspace = 0.75)
plt.suptitle("Before", fontsize = 15)
plt.show()

# We see that there are ouliers present
# so we go for winsorization technique
winsor = Winsorizer(capping_method = 'iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables = list(numeric_features))

data[numeric_features] = winsor.fit_transform(data[numeric_features])

data[numeric_features].plot(kind = 'box', subplots = True, sharey = False, figsize = (14, 8), fontsize = 18)
plt.subplots_adjust(wspace = 0.75)
plt.suptitle("After", fontsize = 15)
plt.show()

# Data PreProcessing

# Define pipelines
num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'mean')), ('scale', MinMaxScaler())])

catget_pipeline = Pipeline([("Encode", OneHotEncoder(drop = "first"))])
# categ_pipeline = Pipeline([('encode', OneHotEncoder(sparse_output = False))])

# Pre - process
process_pipeline = ColumnTransformer([('numeric', num_pipeline, numeric_features),
                                      ('categ', catget_pipeline, categorical_features)],
                                      remainder = 'passthrough',sparse_threshold=0)
# Processed
processed = process_pipeline.fit(X)

# Data Clean
data_clean = pd.DataFrame(processed.transform(X), columns = processed.get_feature_names_out())


data_clean.head()

# Graphical Representation
sns.pairplot(data)
plt.show()

data_corr = data_clean.corr()

# Heat map
#sns.heatmap(data_corr, annot = True, cmap= 'tab20b')

# Here we have output data with imbalance classes so we are going to use stratify to split the data as equal propotion
Y.value_counts() / len(Y)

# Split the data_clean into train and test
x_train, x_test, y_train , y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state = 0)

x_train.shape
y_train.shape
x_test.shape
y_test.shape

y_train.value_counts() / len(y_train)
y_test.value_counts() / len(y_test)

# Model Building (Classification Model)
# Navie Bayses Model 
