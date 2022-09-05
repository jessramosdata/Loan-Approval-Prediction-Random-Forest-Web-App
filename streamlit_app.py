#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import streamlit as st
from IPython import get_ipython

train = pd.read_csv('train.csv')

train = train.dropna()

data['TotalApplicantIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

data = pd.get_dummies(data, columns = ['Gender'], drop_first = True)

data = pd.get_dummies(data, columns = ['Married'], drop_first = True)

data = pd.get_dummies(data, columns = ['Loan_Status'], drop_first = True)
data = data.rename(columns = {'Loan_Status_Y' : 'Loan_Approved'})

train.astype({'Credit_History' : int})

#independent variables
features = ['Gender_Male', 'Married_Yes', 'TotalApplicantIncome', 'LoanAmount', 'Credit_History']
X = data[features]
#dependent variables
dependent = 'Loan_Approved'
y = data[dependent]
X.shape, y.shape

#split dataset into train (80%) and test (20%), shuffle observations
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10, shuffle = True)

#build random forest model, limit max depth to avoid overfitting
forest = RandomForestClassifier(max_depth=4, random_state = 10, n_estimators = 100, min_samples_leaf=5) 
model = forest.fit(x_train, y_train)


@st.cache()

# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History):   
    
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1
 
    if Married == "Unmarried":
        Married = 0
    else:
        Married = 1
 
    if Credit_History == "No Credit History":
        Credit_History = 0
    else:
        Credit_History = 1  
 
    LoanAmount = LoanAmount / 1000
 
    # Making predictions 
    pred_inputs = model.predict(pd.DataFrame([[Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History]]))
        
    if pred_inputs[0] == 0:
        pred = 'I am sorry, you have been rejected for the loan.'
    elif pred_inputs[0] == 1:
        pred = 'Congrats! You have been approved for the loan!'
    else:
        pred = 'Error'
    return pred

def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:teal;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Gender = st.selectbox('Gender',("Male","Female"))
    Married = st.selectbox('Marital Status',("Unmarried","Married")) 
    ApplicantIncome = st.number_input("Total Monthly Income, (Include Coborrower if Applicable)") 
    LoanAmount = st.number_input("Loan Amount (ex. 125000)")
    Credit_History = st.selectbox('Credit History',("Has Credit History","No Credit History"))
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
        st.success('Final Decision: {}'.format(result))
        print(LoanAmount)
     
if __name__=='__main__': 
    main()


# In[ ]:




