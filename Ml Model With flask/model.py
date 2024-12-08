import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  #text to numerical
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

df=pd.read_csv("mail_data.csv")
# print(df)

data=df.where(pd.notnull(df),'')  #fill with True False

data.loc[data['Category'] == 'spam','Category'] = 0 #put 0 where spam
data.loc[data['Category'] == 'ham','Category'] = 1 #put 1 where ham


x=data['Message']
y=data['Category']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3)



feature_extraction = TfidfVectorizer(min_df =1 ,stop_words = 'english' , lowercase=True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')


model = LogisticRegression()


model.fit(x_train_features , y_train)

prediction_on_training_data = model.predict(x_train_features)
accuracy_on_train_data = accuracy_score(y_train , prediction_on_training_data)

prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)

# print("Model Accuracy",accuracy_on_test_data)

input_your_mail = ["""
Subject: Congratulations! You’ve won a $1,000 gift card!

Hello valued customer,

You have been selected to receive a $1,000 gift card from us as part of our loyalty program! Click the link below to claim your prize now:

[Claim your $1,000 gift card now]

Hurry! This offer is valid only for a limited time.

Best regards,
The Rewards Team
"""]

input_data_features = feature_extraction.transform(input_your_mail)

pred = model.predict(input_data_features)

# print("Model pred ",pred)

# if(pred[0]==1):
#   print("ham")
# else:
#   print("spam")

with open('modelStore.pkl', 'wb') as f:
    pickle.dump(model,f)

pickle.dump(feature_extraction,open("fetureExtraction.pkl",'wb'))
