import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import pickle

train = pd.read_csv('DataSet/train.csv')
#test = pd.read_csv('DataSet/test.csv')
y = train['target']
train = train.drop(['id','target'], axis=1)

# Function to one-hot-encode categorical features
def ohe(train, cat_features):
    
    encoder=ce.OneHotEncoder(cols=cat_features,
                             handle_unknown='return_nan',
                             return_df=True,
                             use_cat_names=True)
    data_encoded = encoder.fit_transform(train)
    
    
    train = data_encoded.iloc[:train.shape[0],:]
    
    return train

all_features = train.columns.values
cat_features = []
for i in train.columns.values:
    if i.endswith('cat'):
        cat_features.append(i)
    else:
        continue

bin_features = []
for i in train.columns.values:
    if i.endswith('bin'):
        bin_features.append(i)
    else:
        continue

calc_features = []
for i in train.columns.values:
    if i.startswith('ps_calc'):
        calc_features.append(i)
    else:
        continue

calc_bin_features = []
for i in calc_features:
    if i.endswith('bin'):
        calc_bin_features.append(i)
    else:
        continue

calc_num_features = list(set(calc_features) - set(calc_bin_features))
num_features_with_calc = list((set(train.columns.values) - set(cat_features)) - set(bin_features))
num_features_wo_calc = list(set(num_features_with_calc) - set(calc_features))

train['ps_car_05_cat'] = train['ps_car_05_cat'].replace(-1, 2)
train['ps_car_03_cat'] = train['ps_car_03_cat'].replace(-1, 2)
train=train.replace(-1,'NaN')

# Replacing null values in continuous columns with mean
mean_imp = SimpleImputer(missing_values='NaN', strategy='mean')
mode_imp = SimpleImputer(missing_values='NaN', strategy='most_frequent')
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()

# Replacing null values in categorical columns with mode
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()
train['ps_ind_02_cat'] = mode_imp.fit_transform(train[['ps_ind_02_cat']]).ravel()
train['ps_ind_04_cat'] = mode_imp.fit_transform(train[['ps_ind_04_cat']]).ravel()
train['ps_ind_05_cat'] = mode_imp.fit_transform(train[['ps_ind_05_cat']]).ravel()
train['ps_car_01_cat'] = mode_imp.fit_transform(train[['ps_car_01_cat']]).ravel()
train['ps_car_02_cat'] = mode_imp.fit_transform(train[['ps_car_02_cat']]).ravel()
train['ps_car_07_cat'] = mode_imp.fit_transform(train[['ps_car_07_cat']]).ravel()
train['ps_car_09_cat'] = mode_imp.fit_transform(train[['ps_car_09_cat']]).ravel()

train.drop(calc_features, axis=1, inplace=True)

train=ohe(train,cat_features)

train_scaler = StandardScaler()
train_scaler.fit(train[num_features_wo_calc])

train[num_features_wo_calc] = train_scaler.transform(train[num_features_wo_calc])

catb = CatBoostClassifier(n_estimators = 200, max_depth=7,learning_rate=0.1)
catb.fit(train, y,verbose=False)

# save the model to disk
filename = 'App/fin_cat_model.sav'
pickle.dump(catb, open(filename, 'wb'))