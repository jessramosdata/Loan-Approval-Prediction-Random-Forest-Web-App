#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import pandas as pd
# import numpy as np
# import sklearn as sk
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier 
# import pickle
# from sklearn.tree import export_graphviz
# #from sklearn.tree import tree
# from dtreeviz.trees import dtreeviz
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, accuracy_score
# import seaborn as sns
# import streamlit as st


# In[48]:


get_ipython().run_cell_magic('writefile', 'app.py', '\ntrain = pd.read_csv(\'train.csv\')\n\ntrain = train.dropna()\n\ntrain[\'TotalApplicantIncome\'] = train[\'ApplicantIncome\'] + train[\'ApplicantIncome\']\n\ngender_dummies = pd.get_dummies(train[\'Gender\'])\ntrain = pd.concat((train, gender_dummies), axis = 1)\ntrain = train.drop([\'Gender\'], axis = 1)\ntrain = train.drop([\'Male\'], axis = 1)\ntrain = train.rename(columns = {\'Female\' : \'Gender\'})\n\nmarried_dummies = pd.get_dummies(train[\'Married\'])\ntrain = pd.concat((train, married_dummies), axis = 1)\ntrain = train.drop([\'Married\'], axis = 1)\ntrain = train.drop([\'No\'], axis = 1)\ntrain = train.rename(columns = {\'Yes\' : \'Married\'})\n\nLoanStatus_dummies = pd.get_dummies(train[\'Loan_Status\'])\ntrain = pd.concat((train, LoanStatus_dummies), axis = 1)\ntrain = train.drop([\'Loan_Status\'], axis = 1)\ntrain = train.drop([\'N\'], axis = 1)\ntrain = train.rename(columns = {\'Y\' : \'Loan_Approved\'})\n\ntrain.astype({\'Credit_History\' : int})\n    \n    \n#independent variables\nfeatures = [\'Gender\', \'Married\', \'TotalApplicantIncome\', \'LoanAmount\', \'Credit_History\']\nX = train[features]\n#dependent variables\ndependent = \'Loan_Approved\'\ny = train[dependent]\n\n#split dataset into train (80%) and test (20%), shuffle observations\nx_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10, shuffle = True)\n    \nmodel = RandomForestClassifier(max_depth=4, random_state = 10, n_estimators = 100) \nmodel.fit(x_train, y_train)\n\n\n#@st.cache()\n\n  \n# defining the function which will make the prediction using the data which the user inputs \ndef prediction(Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History):   \n    \n    if Gender == "Male":\n        Gender = 0\n    else:\n        Gender = 1\n \n    if Married == "Unmarried":\n        Married = 0\n    else:\n        Married = 1\n \n    if Credit_History == "No Credit History":\n        Credit_History = 0\n    else:\n        Credit_History = 1  \n \n    LoanAmount = LoanAmount / 1000\n \n    # Making predictions \n    pred_inputs = model.predict([[Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History]])\n        \n        \n    if prediction == 0:\n        pred = \'I am sorry, you have been rejected for the loan.\'\n    else:\n        pred = \'Congrats! You have been approved for the loan!\'\n    return pred\n      \n  \n# this is the main function in which we define our webpage  \ndef main():       \n    # front end elements of the web page \n    html_temp = """ \n    <div style ="background-color:yellow;padding:13px"> \n    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> \n    </div> \n    """\n      \n    # display the front end aspect\n    st.markdown(html_temp, unsafe_allow_html = True) \n      \n    # following lines create boxes in which user can enter data required to make prediction \n    Gender = st.selectbox(\'Gender\',("Male","Female"))\n    Married = st.selectbox(\'Marital Status\',("Unmarried","Married")) \n    ApplicantIncome = st.number_input("Total Monthly Income, (Include Coborrower if Applicable)") \n    LoanAmount = st.number_input("Loan Amount")\n    Credit_History = st.selectbox(\'Credit History\',("Has Credit History","No Credit History"))\n    result =""\n      \n    # when \'Predict\' is clicked, make the prediction and store it \n    if st.button("Predict"): \n        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) \n        st.success(\'Final Decision: {}\'.format(result))\n        print(LoanAmount)\n     \nif __name__==\'__main__\': \n    main()')


# In[47]:



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





# In[ ]:




