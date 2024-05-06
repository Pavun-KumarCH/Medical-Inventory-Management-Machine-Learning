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
from sqlalchemy import create_engine
from feature_engine.outliers import Winsorizer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

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

# Checking for any null values 
data.isna().sum()

# Categories in each column
columns = data.columns

for i in columns:
    classes = data[i].value_counts()
    print("\n",classes)
    
# Drop Irrelavent columns
data.drop(['Patient_ID','RtnMRP','Dateofbill','ReturnQuantity'], axis = True, inplace = True)

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


