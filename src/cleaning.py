"""This code will take raw data of cars, clean it and save the cleaned data. For more info about the cleaning process see 'notebooks/01_data_cleaning.ipynb'"""

import pandas as pd
from sklearn.model_selection import train_test_split

file_loc = "data/raw/Car details v3.csv"
df = pd.DataFrame()
df = pd.read_csv(file_loc)
#The data source is the following https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

df = df.drop_duplicates()
df[pd.isnull(df).any(axis=1)]
#These 209 rows have null values for muliple columns, so we will drop them
df=df.dropna()
units_features = ["engine", "mileage", "max_power"]

cond1 = df["mileage"].apply(lambda string : string.split(" ")[1]) == "km/kg"   
cond2 = (df["fuel"] == "CNG") | (df["fuel"] == "LPG")
#There are only 1% of such values. The representation is too little. We will drop those.
#This is only one percent of data, we can drop corresponding rows

df=df.drop(df[cond1].index)

def drop_units(text : str) -> float :
    """Takes a value with units (eg. 125 bhp). Strips the units part and outputs the value"""

    return float(text.split(" ")[0])


feature_units = {"mileage" : "kmpl", "engine" : "bhp"}  #have units kmpl and bhp

for feature in feature_units.keys():
    df[f"{feature}_{feature_units[feature]}"] = df[feature].apply(drop_units)
    df = df.drop(feature, axis=1)





#There are too many different types of units. Some have range and some dont. Some torque values are in NM whereas other are in kgm. For some rows, one of torque or rpm is missing. Therefore, as a group we decided to drop the column. One could extract reasonable information, but that is a separate project on its own.
df = df.drop("torque", axis = 1)
df["brand"]=df["name"].str.split().str[0]
df["brand"]=df["brand"].replace({"Land":"Land Rover"})

#We can merge some categorical values with other. For example: we can merge dealer and trustmark dealer. I suggest we do this step in EDA/modeling after the data split. 
#Knowing that the data was collected in 2020, a variable Age is added by subtracting the variabe year fron 2020.
df['age']=2020-df['year']
df.drop("year", axis=1)


target_loc_train = "data/processed/Cat_details_v3_cleaned_train.csv"
target_loc_test = "data/processed/Cat_details_v3_cleaned_test.csv"


df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

df_train.to_csv(target_loc_train)
df_test.to_csv(target_loc_test)



print(df.info())
print("Train and test files saved at data/processed")
print(f"Train size: {len(df_train)}")
print(f"Test size: {len(df_test)}")
print(f"Total size: {len(df)}")