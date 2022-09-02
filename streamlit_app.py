#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import sklearn as sk
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import pickle
from sklearn.tree import export_graphviz
from sklearn.tree import tree
from dtreeviz.trees import dtreeviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import streamlit as st


# In[34]:


# train = pd.read_csv('train.csv')

# train = train.dropna()

# train['TotalApplicantIncome'] = train['ApplicantIncome'] + train['ApplicantIncome']

# gender_dummies = pd.get_dummies(train['Gender'])
# train = pd.concat((train, gender_dummies), axis = 1)
# train = train.drop(['Gender'], axis = 1)
# train = train.drop(['Male'], axis = 1)
# train = train.rename(columns = {'Female' : 'Gender'})

# married_dummies = pd.get_dummies(train['Married'])
# train = pd.concat((train, married_dummies), axis = 1)
# train = train.drop(['Married'], axis = 1)
# train = train.drop(['No'], axis = 1)
# train = train.rename(columns = {'Yes' : 'Married'})

# LoanStatus_dummies = pd.get_dummies(train['Loan_Status'])
# train = pd.concat((train, LoanStatus_dummies), axis = 1)
# train = train.drop(['Loan_Status'], axis = 1)
# train = train.drop(['N'], axis = 1)
# train = train.rename(columns = {'Y' : 'Loan_Approved'})

# train.astype({'Credit_History' : int})


# In[35]:


get_ipython().run_cell_magic('writefile', 'app.py', ' \nimport pickle\nimport streamlit as st\n \n# loading the trained model\npickle_in = open(\'Loan_Approval_Prediction.pkl\', \'rb\') \nclassifier = pickle.load(pickle_in)\n\n@st.cache()\n\n  \n# defining the function which will make the prediction using the data which the user inputs \ndef prediction(Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History):   \n    \n    if Gender == "Male":\n        Gender = 0\n    else:\n        Gender = 1\n \n    if Married == "Unmarried":\n        Married = 0\n    else:\n        Married = 1\n \n    if Credit_History == "No Credit History":\n        Credit_History = 0\n    else:\n        Credit_History = 1  \n \n    LoanAmount = LoanAmount / 1000\n \n    # Making predictions \n    pred_inputs = model.predict([[Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History]])\n        \n        \n    if prediction == 0:\n        pred = \'I am sorry, you have been rejected for the loan.\'\n    else:\n        pred = \'Congrats! You have been approved for the loan!\'\n    return pred\n      \n  \n# this is the main function in which we define our webpage  \ndef main():       \n    # front end elements of the web page \n    html_temp = """ \n    <div style ="background-color:yellow;padding:13px"> \n    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> \n    </div> \n    """\n      \n    # display the front end aspect\n    st.markdown(html_temp, unsafe_allow_html = True) \n      \n    # following lines create boxes in which user can enter data required to make prediction \n    Gender = st.selectbox(\'Gender\',("Male","Female"))\n    Married = st.selectbox(\'Marital Status\',("Unmarried","Married")) \n    ApplicantIncome = st.number_input("Total Monthly Income, (Include Coborrower if Applicable)") \n    LoanAmount = st.number_input("Loan Amount")\n    Credit_History = st.selectbox(\'Credit History\',("Has Credit History","No Credit History"))\n    result =""\n      \n    # when \'Predict\' is clicked, make the prediction and store it \n    if st.button("Predict"): \n        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) \n        st.success(\'Final Decision: {}\'.format(result))\n        print(LoanAmount)\n     \nif __name__==\'__main__\': \n    main()')


# In[24]:



# #original

# %%writefile app.py
 
# import pickle
# import streamlit as st
 
# # loading the trained model
# pickle_in = open('classifier.pkl', 'rb') 
# classifier = pickle.load(pickle_in)
 
# @st.cache()
  
# # defining the function which will make the prediction using the data which the user inputs 
# def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
 
#     # Pre-processing user input    
#     if Gender == "Male":
#         Gender = 0
#     else:
#         Gender = 1
 
#     if Married == "Unmarried":
#         Married = 0
#     else:
#         Married = 1
 
#     if Credit_History == "Unclear Debts":
#         Credit_History = 0
#     else:
#         Credit_History = 1  
 
#     LoanAmount = LoanAmount / 1000
 
#     # Making predictions 
#     prediction = classifier.predict( 
#         [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])
     
#     if prediction == 0:
#         pred = 'Rejected'
#     else:
#         pred = 'Approved'
#     return pred
      
  
# # this is the main function in which we define our webpage  
# def main():       
#     # front end elements of the web page 
#     html_temp = """ 
#     <div style ="background-color:yellow;padding:13px"> 
#     <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
#     </div> 
#     """
      
#     # display the front end aspect
#     st.markdown(html_temp, unsafe_allow_html = True) 
      
#     # following lines create boxes in which user can enter data required to make prediction 
#     Gender = st.selectbox('Gender',("Male","Female"))
#     Married = st.selectbox('Marital Status',("Unmarried","Married")) 
#     ApplicantIncome = st.number_input("Applicants monthly income") 
#     LoanAmount = st.number_input("Total loan amount")
#     Credit_History = st.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))
#     result =""
      
#     # when 'Predict' is clicked, make the prediction and store it 
#     if st.button("Predict"): 
#         result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
#         st.success('Your loan is {}'.format(result))
#         print(LoanAmount)
     
# if __name__=='__main__': 
#     main()
    


# In[ ]:


#independent variables
features = ['Gender', 'Married', 'TotalApplicantIncome', 'LoanAmount', 'Credit_History']
X = train[features]
#dependent variables
dependent = 'Loan_Approved'
y = train[dependent]

#split dataset into train (80%) and test (20%), shuffle observations
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10, shuffle = True)
    
model = RandomForestClassifier(max_depth=4, random_state = 10, n_estimators = 100) 
model.fit(x_train, y_train)

