import pandas as pd
import numpy as np
from helper.preprocessing import CategoricalFeatures
from sklearn.preprocessing import normalize

def feature_engineering(df):
    # rename column name
    cars_data.rename(
    columns=({"dateCreated": "ad_created",
              "dateCrawled": "date_crawled",
              "fuelType": "fuel_type",
              "lastSeen": "last_seen",
              "monthOfRegistration": "registration_month",
              "notRepairedDamage": "unrepaired_damage",
              "nrOfPictures": "num_of_pictures",
              "offerType": "offer_type",
              "postalCode": "postal_code",
              "powerPS": "power_ps",
              "vehicleType": "vehicle_type",
              "yearOfRegistration": "registration_year"}),
    inplace=True)
    
    # change data type
    cars_data["ad_created"]=pd.to_datetime(cars_data["ad_created"])
    cars_data["date_crawled"]=pd.to_datetime(cars_data["date_crawled"])
    cars_data["last_seen"]=pd.to_datetime(cars_data["last_seen"])
    
    # string replace
    cars_data['price'] = cars_data['price'].str.replace(',', '')
    cars_data['price'] = cars_data['price'].str.replace('$', '')
    cars_data['price'] = cars_data['price'].astype('int64')

    cars_data['odometer'] = cars_data['odometer'].str.replace(',', '')
    cars_data['odometer'] = cars_data['odometer'].str.replace('km', '')
    cars_data['odometer'] = cars_data['odometer'].astype('int64')
    cars_data.rename(columns=({"odometer": "odometer_km"}), inplace=True)
    
    # drop columns
    cars_data.drop('num_of_pictures', axis=1, inplace=True)
    cars_data.drop(['name','postal_code'], axis=1, inplace=True)
    
    # delete rows out of range 500 to 40000
    cars_data.drop(cars_data['price'][~cars_data['price'].between(500,40000)].index, inplace=True)

    # fill NA data
    categorical = cars_data.loc[:,[col for col in cars_data.columns if cars_data[col].dtypes == 'object']]
    for a in categorical:
        data_clean = cars_data[a].fillna(cars_data[a].mode()[0], inplace=True)
    numerical = cars_data.loc[:,[col for col in cars_data.columns if cars_data[col].dtypes == 'int64']]
    for b in numerical:
        data_clean = cars_data[b].fillna(cars_data[b].median(), inplace=True)
        
    #normalize data
    cols_normalize = ['registration_year','power_ps','odometer_km','registration_month']
    cars_data[cols_normalize] = normalize(X=cars_data[cols_normalize], norm="l2", axis=1)
    
    # encoding numerical data
    cols_encoder = [col for col in cars_data.columns if cars_data[col].dtypes == 'object']
    label_enc = preprocessing.LabelEncoder()
    for i in cols_encoder:
        label_enc.fit(cars_data[i].values)
        cars_data[i] = label_enc.transform(cars_data[i].values)

    # applied CategoricalFeatures()
    cols = [c for c in df_transformed.columns if (df_transformed[c].dtype != "float" and df_transformed[c].dtype != "int")]
    cat_feats = CategoricalFeatures(df_transformed, categorical_features=cols, encoding_type="one_hot", handle_na=True)
    df_transformed = cat_feats.fit_transform()

    return df_transformed