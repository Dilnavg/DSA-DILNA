#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


media=pd.read_csv(r'C:\DSA\mediacompany.csv')


# In[4]:


media.head()


# In[5]:


media.shape


# In[6]:


sum(media.duplicated(subset='Date'))==0


# In[7]:


media.columns


# In[8]:


#data exploration


# In[9]:


media.shape


# In[10]:


media.info()


# In[11]:


media.describe()


# In[12]:


media.isna().sum()


# In[13]:


#outlier detection


# In[20]:


fig,ax=plt.subplots(2,2,figsize=(10,6))
plt1=sns.boxplot(media['Views_show'],ax=ax[0,0])
plt2=sns.boxplot(media['Visitors'],ax=ax[0,1])
plt3=sns.boxplot(media['Views_platform'],ax=ax[1,0])
plt4=sns.boxplot(media['Ad_impression'],ax=ax[1,1])


# In[21]:


media['Date']=pd.to_datetime(media['Date'],dayfirst=False)


# In[22]:


media.head()


# In[23]:


media['Day']=media['Date'].dt.dayofweek


# In[24]:


media.head()


# In[25]:


media.plot.line(x='Date',y='Views_show')


# In[26]:


sns.barplot(x='Day',y='Views_show',data=media)


# In[27]:


#saturdy and sunday is more show


# In[28]:


status={5:1,6:1,0:0,1:0,2:0,3:0,4:0}
media['weekend']=media['Day'].map(status)


# In[29]:


media.head()


# In[30]:


sns.barplot(data=media,x='weekend',y='Views_show')


# In[31]:


#weekends have higher number of shows


# In[34]:


x1=media.plot(x='Date',y='Views_show')
x2=x1.twinx()
media.plot(x='Date',y='Ad_impression',ax=x2,color='red',legend=False)
x1.figure.legend()


# In[35]:


sns.scatterplot(data=media,x='Views_platform',y='Views_show')


# In[38]:


sns.barplot(x='Cricket_match_india',y='Views_show',data=media)


# In[39]:


sns.barplot(x='Character_A',y='Views_show',data=media)


# # model creation

# In[41]:


#min max scaler


# In[43]:


from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()


# In[44]:


num_cols=['Views_show','Visitors', 'Views_platform', 'Ad_impression']
media[num_cols]=scale.fit_transform(media[num_cols])


# In[45]:


media.head()


# In[46]:


sns.heatmap(media.corr(),annot=True)


# In[47]:


x=media[['Visitors','weekend']]
y=media['Views_show']


# In[48]:


from sklearn.linear_model import LinearRegression
ln=LinearRegression()
ln.fit(x,y)


# In[56]:


import statsmodels.api as sm
x=sm.add_constant(x)
lm1=sm.OLS(y,x).fit()
print(lm1.summary())


# In[57]:


x=media[['Visitors','weekend','Character_A']]
y=media['Views_show']


# In[58]:


x=sm.add_constant(x)
lm2=sm.OLS(y,x).fit()
print(lm2.summary())


# In[59]:


media['Lag_view']=np.roll(media['Views_show'],1)
media.head()


# In[60]:


#in case of series viewing lagview is added


# In[61]:


x=media[['Visitors','weekend','Lag_view']]
y=media['Views_show']


# In[62]:


x=sm.add_constant(x)
lm3=sm.OLS(y,x).fit()
print(lm3.summary())


# In[63]:


x=media[['Visitors','weekend','Character_A','Ad_impression']]
y=media['Views_show']


# In[64]:


x=sm.add_constant(x)
lm4=sm.OLS(y,x).fit()
print(lm4.summary())


# In[65]:


predicted_view=lm4.predict(x)


# In[66]:


from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(media.Views_show,predicted_view)
r_sqr=r2_score(media.Views_show,predicted_view)


# In[67]:


print('MSE :',mse)
print('R square value :',r_sqr)


# In[70]:


s=[i for i in range(1,81,1)]
fig=plt.figure()
plt.plot(s,media.Views_show,color='green',linestyle='-')
plt.plot(s,predicted_view,color='red',linestyle='-')
plt.xlabel('Index')
plt.ylabel('Viewscount')


# In[72]:


s=[i for i in range(1,81,1)]
fig=plt.figure()
plt.plot(s,media.Views_show-predicted_view,color='blue',linestyle='-')

plt.title('Error',fontsize=22)
plt.xlabel('Index')
plt.ylabel('Views-predicted')


# In[ ]:




