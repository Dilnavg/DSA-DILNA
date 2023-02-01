#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[107]:


data=pd.read_csv(r'C:\DSA\income_evaluation.csv')


# In[108]:


data.head


# In[109]:


data.columns


# In[110]:


data.shape


# In[111]:


data.corr()


# In[112]:


sns.heatmap(data.corr())


# In[113]:


data['workclass'].unique()


# In[114]:


data['native-country'].unique()


# In[115]:


data['occupation'].unique()


# In[116]:


data[['workclass','native-country','occupation']]=data[['workclass','native-country','occupation']].replace('?',np.NaN)


# In[117]:


data['workclass'].unique()


# In[118]:


data['native-country'].unique()


# In[119]:


data.isna().sum()


# In[120]:


data.iloc[69,:]


# In[121]:


data=data.fillna(data.mode().iloc[0])


# In[122]:


plt.figure(figsize=(10,8))


# In[123]:


x=sns.countplot(x='income',data=data)
x.set_title('distribution of income feature')
plt.show()


# In[124]:


plt.figure(figsize=(10,8))
x=sns.countplot(x='income',data=data,hue='sex')
x.set_title('distribution of income feature wrt gender')
plt.show()


# In[125]:


plt.figure(figsize=(10,8))
x=data.workclass.value_counts().plot(kind='bar',color='blue',legend=True)
x.set_title('distribution of workclass feature')
plt.show()


# In[126]:


plt.figure(figsize=(10,8))
x=sns.countplot(x='workclass',hue='income',data=data)
x.set_title('distribution of workclass feature wrt income')
x.legend(loc='upper right')
plt.show()


# In[127]:


#most of them are private sector with <50k salary


# In[128]:


data['income'].unique()


# In[129]:


data['income']=data['income'].replace(('<=50K', '>50K'),(0,1))


# In[130]:


data['income']


# In[131]:


#duplicate values check


# In[132]:


data.drop_duplicates(inplace=True)


# In[133]:


data.shape


# In[134]:


data.head()


# In[135]:


data.drop(['education'],axis=1,inplace=True)


# In[136]:


data.head()


# In[137]:


#encoding of data


# In[138]:


data=pd.get_dummies(data)


# In[139]:


data.head()


# In[140]:


data.corr()


# In[141]:


correlation=data.corr()['income'].sort_values(ascending=False)
print('highly positive correlation \n',correlation.head(15))
print('highly negative correlation \n',correlation.tail(15))


# In[142]:


data.columns


# In[143]:


x=data.drop('income',axis=1)


# In[145]:


x


# In[146]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=.25)


# In[147]:


#standard scaler


# In[149]:


from sklearn.preprocessing import StandardScaler


# In[151]:


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[152]:


#models


# In[153]:


from sklearn.linear_model import LogisticRegression


# In[155]:


lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[156]:


y_prediction=lr.predict(x_test)
y_prediction


# In[157]:


from sklearn.metrics import *


# In[160]:


cm=confusion_matrix(y_test,y_prediction)


# In[164]:


plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True)
plt.show()


# In[ ]:




