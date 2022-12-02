#import package
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#import the data
uploaded_file = st.file_uploader("saleprice.csv")
train = pd.read_csv("saleprice.csv")
st.title("Predict Your Dream House Here!")

#pick variables
train_a=train[['OverallQual', 'GrLivArea', 'GarageCars', 'SalePrice']].copy()
train_a.info()

# Pembagian X dan y
X = train_a.drop('SalePrice',axis=1)
y= train_a['SalePrice']


## Splitting X and Y into training and testing data
from sklearn.model_selection import train_test_split
X_train, y_test, y_train, X_test = train_test_split(X,y, test_size=0.4,
                                                    random_state=1)

#input the numbers
OverallQual = st.number_input("Overall Quality",int(X_train.OverallQual.min()),int(X_train.OverallQual.max()),int(X_train.OverallQual.mean()))
GrLivArea = st.number_input("Grade Living Area",int(train_a.GrLivArea.min()),int(train_a.GrLivArea.max()),int(train_a.GrLivArea.mean()))
GarageCars = st.number_input("Garage Cars",int(train_a.GarageCars.min()),int(train_a.GarageCars.max()),int(train_a.GarageCars.mean()))


## Create linear regression object
from sklearn import linear_model
reg = linear_model.LinearRegression()

# Train the model using the traing sets
reg.fit(X_train, y_train)

#Output
predictions = reg.predict([[OverallQual,GrLivArea,GarageCars]])[0]

if st.button('Predict'):
    st.header('Your estimated house price will be $ {}'.format(int(predictions)))