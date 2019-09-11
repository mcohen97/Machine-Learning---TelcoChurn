from C45 import C45 
import pandas
import numpy
import sklearn

dataframe = pandas.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
dataframe['tenure'] = dataframe['tenure'].astype(float)
dataframe = dataframe.dropna(axis = 0)
#dataframe = dataframe.head(100)
#dataframe = pandas.read_csv("Tennis_Data.csv")


c_gender = 'gender' 
c_senior_citizen = 'SeniorCitizen'
c_partner = 'Partner'
c_dependents = 'Dependents'
c_tenure = 'tenure'
c_phone_service = 'PhoneService'
c_multiple_lines ='MultipleLines'
c_internet_service = 'InternetService'
c_online_security = 'OnlineSecurity'
c_online_backup = 'OnlineBackup'
c_device_protection ='DeviceProtection'
c_tech_support ='TechSupport'
c_streaming_tv = 'StreamingTV'
c_streaming_moves = 'StreamingMovies'
c_contract ='Contract'
c_paperless_billing ='PaperlessBilling'
c_payment_method = 'PaymentMethod'
c_monthly_charges ='MonthlyCharges'
c_total_charges = 'TotalCharges'
c_churn ='Churn'

categorical_features = [c_gender, c_senior_citizen, c_partner, c_dependents, c_phone_service,
            c_multiple_lines,c_internet_service,c_online_security, c_online_backup, c_device_protection,
            c_tech_support, c_streaming_tv,c_streaming_moves, c_contract, c_paperless_billing,c_payment_method]

numeric_features = [c_tenure, c_monthly_charges, c_total_charges]


'''
c_day = 'Day'	
c_outlook = 'Outlook' 	
c_temperature = 'Temperature' 	
c_humidity = 'Humidity'
c_wind = 'Wind' 
c_play_tennis = 'PlayTennis'

categorical_features = [
    #c_day, 
    c_outlook, c_temperature, c_humidity, c_wind]
numeric_features = []
'''

from sklearn.preprocessing import LabelEncoder

encoded_features = []

for feature in categorical_features:
    label_encoder_contract = LabelEncoder()
    label_encoder_contract.fit(dataframe[feature])
    dataframe[feature+'_encoded'] = label_encoder_contract.transform(dataframe[feature])
    encoded_features.append(feature+'_encoded')
    
predictor_features = encoded_features + numeric_features

#proyeccion_data = dataframe[[c_gender,c_gender+'_encoded' ]]

dataframe[c_total_charges] = pandas.to_numeric(dataframe[c_total_charges], errors='coerce')

dataframe.dropna(axis = 0, inplace = True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataframe[predictor_features],dataframe[c_churn], test_size = 0.33)
#X_train, X_test, y_train, y_test = train_test_split(dataframe[predictor_features],dataframe[c_play_tennis], test_size = 0.33)
#X_train = dataframe[predictor_features]
#y_train = dataframe[c_play_tennis]


tree = C45()
tree.fit(X_train, y_train, 5)
y_predict = tree.predict(X_test)
print (y_predict)