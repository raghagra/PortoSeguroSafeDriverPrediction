from flask import Flask, render_template, make_response, request
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
import category_encoders as ce
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler


app=Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

@app.route("/", methods = ["GET"])
def home():
    return render_template("index.html")



@app.route("/predict", methods = ["POST"])
def bulk_pred():
    if request.method == "POST":
        f = request.files['data_file']
        if not f:
            return "No file"
        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream)
        print(csv_input)
        for row in csv_input:
            print(row)

    stream.seek(0)
    result = transform(stream.read())
    train = pd.read_csv('train.csv')

    df = pd.read_csv(StringIO(result))
    df_org=df.copy()
    calc_features=['ps_calc_01', 'ps_calc_02', 'ps_calc_03','ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12','ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']
    df.drop(calc_features, axis=1, inplace=True)
    train.drop(calc_features, axis=1, inplace=True)

    # load the model from disk
    filename = 'fin_cat_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    # load scalers from the disk
    filename = 'scaler_dump.sav'
    scaler_dump = pickle.load(open(filename, 'rb'))
    # load encoders from the disk
    filename = 'encoder_dump.sav'
    encoder_dump = pickle.load(open(filename, 'rb'))

    # load mode imputers
 
    Imputer_ps_car_11 = 'Imputer_ps_car_11.sav'
    Imputer_ps_car_11 = pickle.load(open(Imputer_ps_car_11, 'rb'))

    Imputer_ps_ind_02_cat = 'Imputer_ps_ind_02_cat.sav'
    Imputer_ps_ind_02_cat = pickle.load(open(Imputer_ps_ind_02_cat, 'rb'))

    Imputer_ps_ind_04_cat = 'Imputer_ps_ind_04_cat.sav'
    Imputer_ps_ind_04_cat = pickle.load(open(Imputer_ps_ind_04_cat, 'rb'))

    Imputer_ps_ind_05_cat = 'Imputer_ps_ind_05_cat.sav'
    Imputer_ps_ind_05_cat = pickle.load(open(Imputer_ps_ind_05_cat, 'rb'))

    Imputer_ps_car_01_cat = 'Imputer_ps_car_01_cat.sav'
    Imputer_ps_car_01_cat = pickle.load(open(Imputer_ps_car_01_cat, 'rb'))

    Imputer_ps_car_02_cat = 'Imputer_ps_car_02_cat.sav'
    Imputer_ps_car_02_cat = pickle.load(open(Imputer_ps_car_02_cat, 'rb'))

    Imputer_ps_car_07_cat = 'Imputer_ps_car_07_cat.sav'
    Imputer_ps_car_07_cat = pickle.load(open(Imputer_ps_car_07_cat, 'rb'))

    Imputer_ps_car_09_cat = 'Imputer_ps_car_09_cat.sav'
    Imputer_ps_car_09_cat = pickle.load(open(Imputer_ps_car_09_cat, 'rb'))

    # load mean imputers
    Imputer_ps_reg_03 = 'Imputer_ps_reg_03.sav'
    Imputer_ps_reg_03 = pickle.load(open(Imputer_ps_reg_03, 'rb'))
    
    Imputer_ps_car_12 = 'Imputer_ps_car_12.sav'
    Imputer_ps_car_12 = pickle.load(open(Imputer_ps_car_12, 'rb'))

    Imputer_ps_car_14 = 'Imputer_ps_car_14.sav'
    Imputer_ps_car_14 = pickle.load(open(Imputer_ps_car_14, 'rb'))

    # transforming the data
    df['ps_car_05_cat'] = df['ps_car_05_cat'].replace(-1, 2)
    df['ps_car_03_cat'] = df['ps_car_03_cat'].replace(-1, 2)
    # df=df.replace(-1,'NaN')

    
    df['ps_reg_03'] = Imputer_ps_reg_03.transform(df[['ps_reg_03']]).ravel()
    df['ps_car_12'] = Imputer_ps_car_12.transform(df[['ps_car_12']]).ravel()
    df['ps_car_14'] = Imputer_ps_car_14.transform(df[['ps_car_14']]).ravel()
    

    # Replacing null values in categorical columns with mode
    df['ps_car_11'] = Imputer_ps_car_11.transform(df[['ps_car_11']]).ravel()
    df['ps_ind_02_cat'] = Imputer_ps_ind_02_cat.transform(df[['ps_ind_02_cat']]).ravel()
    df['ps_ind_04_cat'] = Imputer_ps_ind_04_cat.transform(df[['ps_ind_04_cat']]).ravel()
    df['ps_ind_05_cat'] = Imputer_ps_ind_05_cat.transform(df[['ps_ind_05_cat']]).ravel()
    df['ps_car_01_cat'] = Imputer_ps_car_01_cat.transform(df[['ps_car_01_cat']]).ravel()
    df['ps_car_02_cat'] = Imputer_ps_car_02_cat.transform(df[['ps_car_02_cat']]).ravel()
    df['ps_car_07_cat'] = Imputer_ps_car_07_cat.transform(df[['ps_car_07_cat']]).ravel()
    df['ps_car_09_cat'] = Imputer_ps_car_09_cat.transform(df[['ps_car_09_cat']]).ravel()

    # ohe on cat variables
    df = encoder_dump.transform(df)
    print(df.columns)

    # scale using normalization
    num_features_wo_calc=['ps_car_11','ps_ind_03','ps_car_14','ps_reg_02','ps_car_12','ps_ind_01','ps_car_13','ps_reg_01','ps_ind_15','ps_car_15','ps_reg_03','ps_ind_14']
    df[num_features_wo_calc] = scaler_dump.transform(df[num_features_wo_calc])

    df_org['prediction']=loaded_model.predict_proba(df)[:,1]

    

    response = make_response(df_org.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response

    
        

    
    

if __name__=="__main__":
    app.run()