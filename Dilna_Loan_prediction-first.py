#!/usr/bin/env python
# coding: utf-8

# In[277]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[278]:


train_data=pd.read_csv(r'C:\DSA\train_ctrUa4K.csv')
test_data=pd.read_csv(r'C:\DSA\test_lAUu6dG.csv')


# In[279]:


train_original=train_data.copy() 
test_original=test_data.copy()


# In[280]:


train_data.head()


# In[281]:


train_data.columns


# In[282]:


test_data.head()


# In[283]:


test_data.columns


# In[284]:


train_data.info()


# In[285]:


test_data.info()


# In[286]:


train_data.shape


# In[287]:


test_data.shape


# In[288]:


#Univariate Analysis- It is the easiest form of analyzing data where we analyze each variable individually.


# In[289]:


#Target variable-Loan_status-As it is a categorical variable, let us look at its frequency table, percentage distribution, and bar plot.


# In[290]:


train_data['Loan_Status'].value_counts()


# In[291]:


train_data['Loan_Status'].value_counts(normalize=True)


# In[292]:


train_data['Loan_Status'].value_counts().plot.bar()


# In[293]:


#422(around 69%) people out of 614 got the approvalof loan


# In[294]:


#Categorical features(independent variable): These features have categories (Gender, Married, Self_Employed, Credit_History, Loan_Status)


# In[295]:


train_data['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 


# In[296]:


train_data['Married'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Married')


# In[297]:


train_data['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Self_Employed')


# In[298]:


train_data['Credit_History'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Credit_History')


# In[299]:


#Most of the  applicants in the dataset are male.
#Most of applicants in the dataset are married.
#Most of the  applicants in the dataset are self-employed.
#Most of the  applicants have repaid their debts.


# In[300]:


#Ordinal features(independent): Variables in categorical features having some order involved (Dependents, Education, Property_Area)


# In[301]:


plt.figure(1)


# In[302]:


plt.subplot(131)
train_data['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6),title='Dependents') 
plt.subplot(132)
train_data['Education'].value_counts(normalize=True).plot.bar(title= 'Education') 
plt.subplot(133) 
train_data['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area') 
plt.show()


# In[303]:


#Most of the applicants don’t have dependents.
#Most of the  applicants are graduates.
#Most of the applicants are from semi-urban areas.


# In[304]:


#Numerical features: These features have numerical values (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)


# In[ ]:





# In[305]:


plt.figure(1)
plt.subplot(111) 
train_data['ApplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()


# In[306]:


plt.figure(1)
plt.subplot(111) 
train_data['LoanAmount'].plot.box(figsize=(16,5)) 
plt.show()


# In[307]:


##Bivariate Analysis categorical vs Target variable


# In[308]:


Gender=pd.crosstab(train_data['Gender'],train_data['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[309]:


Married=pd.crosstab(train_data['Married'],train_data['Loan_Status']) 
Dependents=pd.crosstab(train_data['Dependents'],train_data['Loan_Status']) 
Education=pd.crosstab(train_data['Education'],train_data['Loan_Status']) 
Self_Employed=pd.crosstab(train_data['Self_Employed'],train_data['Loan_Status']) 
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(4,4)) 
plt.show() 
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show() 
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(4,4)) 
plt.show() 
Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot(kind="bar",figsize=(4,4))
plt.show()


# In[310]:


Credit_History=pd.crosstab(train_data['Credit_History'],train_data['Loan_Status']) 
Property_Area=pd.crosstab(train_data['Property_Area'],train_data['Loan_Status']) 
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show()


# In[311]:


#Numerical Independent Variable vs Target Variable
# find the mean income of people for which the loan has been approved vs the mean income of people for which the loan has not been approved.


# In[312]:


train_data.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


# In[313]:


bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train_data['Income_bin']=pd.cut(train_data['ApplicantIncome'],bins,labels=group)


# In[314]:


Income_bin=pd.crosstab(train_data['Income_bin'],train_data['Loan_Status']) 
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('ApplicantIncome') 
P = plt.ylabel('Percentage')


# In[315]:


#It can be inferred that Applicant’s income does not affect the chances of loan approval


# In[316]:


bins=[0,1000,3000,42000]
group=['Low','Average','High'] 
train_data['Coapplicant_Income_bin']=pd.cut(train_data['CoapplicantIncome'],bins,labels=group)


# In[317]:


Coapplicant_Income_bin=pd.crosstab(train_data['Coapplicant_Income_bin'],train_data['Loan_Status']) 
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('CoapplicantIncome') 
P = plt.ylabel('Percentage')


# In[318]:


train_data['Total_Income']=train_data['ApplicantIncome']+train_data['CoapplicantIncome']


# In[319]:


bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train_data['Total_Income_bin']=pd.cut(train_data['Total_Income'],bins,labels=group)


# In[320]:


Total_Income_bin=pd.crosstab(train_data['Total_Income_bin'],train_data['Loan_Status']) 
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('Total_Income') 
P = plt.ylabel('Percentage')


# In[321]:


#We can see that Proportion of loans getting approved for applicants having low Total_Income is very less as compared to that of applicants with Average, High, and Very High Income.


# In[322]:


bins=[0,100,200,700] 
group=['Low','Average','High'] 
train_data['LoanAmount_bin']=pd.cut(train_data['LoanAmount'],bins,labels=group)


# In[323]:


LoanAmount_bin=pd.crosstab(train_data['LoanAmount_bin'],train_data['Loan_Status']) 
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('LoanAmount') 
P = plt.ylabel('Percentage')


# In[324]:


# the chances of loan approval will be high when the loan amount is less.


# In[325]:


train_data=train_data.drop(['Income_bin', 'Coapplicant_Income_bin',
 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)


# In[326]:


train_data['Dependents'].replace('3+', 3,inplace=True) 
train_data['Dependents'].replace('3+', 3,inplace=True) 
train_data['Loan_Status'].replace('N', 0,inplace=True) 
train_data['Loan_Status'].replace('Y', 1,inplace=True)


# In[327]:


corr_matrix=train_data.corr()
sns.heatmap(corr_matrix,annot =True,cmap='YlGnBu')
plt.plot()


# In[328]:


#the most correlated variables are (ApplicantIncome – LoanAmount) and (Credit_History – Loan_Status). LoanAmount is also correlated with CoapplicantIncome.


# # Missing value outlier

# In[329]:


train_data.isnull().sum()


# In[330]:


train_data['Gender'].fillna(train_data['Gender'].mode()[0], inplace=True)
train_data['Married'].fillna(train_data['Married'].mode()[0], inplace=True)
train_data['Dependents'].fillna(train_data['Dependents'].mode()[0], inplace=True)
train_data['Self_Employed'].fillna(train_data['Self_Employed'].mode()[0], inplace=True)
train_data['Credit_History'].fillna(train_data['Credit_History'].mode()[0], inplace=True)


# In[331]:


train_data.isnull().sum()


# In[332]:


train_data['Loan_Amount_Term'].value_counts()


# In[333]:


train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mode()[0], inplace=True)


# In[334]:


train_data['LoanAmount'].fillna(train_data['LoanAmount'].median(), inplace=True)


# In[335]:


train_data.isnull().sum()


# # missing values in test dataset

# In[336]:


test_data.isna().sum()


# In[337]:


test_data['Gender'].fillna(test_data['Gender'].mode()[0], inplace=True) 
test_data['Dependents'].fillna(test_data['Dependents'].mode()[0], inplace=True) 
test_data['Self_Employed'].fillna(test_data['Self_Employed'].mode()[0], inplace=True) 
test_data['Credit_History'].fillna(test_data['Credit_History'].mode()[0], inplace=True) 
test_data['Loan_Amount_Term'].fillna(test_data['Loan_Amount_Term'].mode()[0], inplace=True) 
test_data['LoanAmount'].fillna(test_data['LoanAmount'].median(), inplace=True)


# In[338]:


# One way to remove the skewness is by doing the log transformation. As we take the log transformation, it does not affect the smaller values much but reduces the larger values. So, we get a distribution similar to the normal distribution.


# In[339]:


train_data['LoanAmount_log'] = np.log(train_data['LoanAmount']) 
train_data['LoanAmount_log'].hist(bins=20) 
test_data['LoanAmount_log'] = np.log(test_data['LoanAmount'])


# # MODEL BUILDING-Logistic Regression

# In[340]:


train_data=train_data.drop('Loan_ID',axis=1) 
test_data=test_data.drop('Loan_ID',axis=1)


# In[341]:


#Sklearn requires the target variable in a separate dataset. So, we will drop our target variable from the training dataset and save it in another dataset.


# In[342]:


X = train_data.drop('Loan_Status',1) 
y = train_data.Loan_Status


# In[343]:


#make dummy variables for the categorical variables. A dummy variable turns categorical variables into a series of 0 and 1, making them a lot easier to quantify and compare. 


# In[344]:


X=pd.get_dummies(X)
train_data=pd.get_dummies(train_data)
test_data=pd.get_dummies(test_data)


# In[345]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[346]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[347]:


model = LogisticRegression() 
model.fit(x_train, y_train)


# In[348]:


pred_cv = model.predict(x_test)


# In[349]:


accuracy_score(y_test,pred_cv)


# In[350]:


#81% of the loan status correctly in train data


# In[351]:


pred_test = model.predict(test_data)


# In[352]:


submission=pd.read_csv(r'C:\DSA\sample_submission.csv')


# In[353]:


#We only need the Loan_ID and the corresponding Loan_Status for the final submission. we will fill these columns with the Loan_ID of the test dataset and the predictions that we made, i.e., pred_test respectively.


# In[354]:


submission['Loan_Status']=pred_test 
submission['Loan_ID']=test_original['Loan_ID']


# In[355]:


#predictions in Y and N. So let’s convert 1 and 0 to Y and N.


# In[356]:


submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)


# In[357]:


submission['Loan_Status']


# In[381]:





# In[358]:


#KNN


# In[359]:


from sklearn.neighbors import KNeighborsClassifier


# In[360]:


metric_k=[]
neighbors=np.arange(3,15)


# In[361]:


for k in neighbors:
    classifier=KNeighborsClassifier(n_neighbors=k,metric='euclidean')
    classifier.fit(x_train,y_train)
    y_prediction=classifier.predict(x_test)
    acc=accuracy_score(y_test,y_prediction)
    metric_k.append(acc)


# In[362]:


plt.plot(neighbors,metric_k,'o-')
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.grid()


# In[363]:


classifier=KNeighborsClassifier(n_neighbors=11,metric='euclidean')
classifier.fit(x_train,y_train)
y_prediction=classifier.predict(x_test)


# In[364]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[365]:


print('Accuracy is:',accuracy_score(y_test,y_prediction))


# In[ ]:





# In[366]:


#Decision tress


# In[367]:


from sklearn.tree import DecisionTreeClassifier
dt_cls=DecisionTreeClassifier()
dt_cls=dt_cls.fit(x_train,y_train)
y_pred_dt=dt_cls.predict(x_test)


# In[368]:


confusion_matrix(y_test,y_pred_dt)


# In[369]:


accuracy_score(y_test,y_pred_dt)


# In[380]:


#Making the final submission


# In[382]:


submission=pd.read_csv(r'C:\DSA\sample_submission.csv')


# In[383]:


submission['Loan_Status']=pred_test 
submission['Loan_ID']=test_original['Loan_ID']


# In[384]:


submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)


# In[388]:


pd.DataFrame(submission,columns=['Loan_ID','Loan_Status']).to_csv('submission.csv')


# In[389]:


from IPython.display import HTML
import base64 
def create_download_link( loancsv, title = "Download CSV file", filename = "data.csv"):  
    csv = loancsv.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(submission)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




