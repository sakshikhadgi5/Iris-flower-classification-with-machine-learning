#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


# In[3]:


data=pd.read_csv("C:\\Users\\ADMIN\\Dropbox\\PC\\Downloads\\Iris.csv")
data.head()


# In[4]:


data.shape


# In[5]:


data['Species'].value_counts()


# In[6]:


data.isna().sum()


# In[7]:


data=data.drop(columns=['Id'])
data.describe()


# In[8]:


encoder=LabelEncoder()
encoder.fit(data['Species'])
data['Species']=encoder.fit_transform(data['Species'])
dictionary=dict(enumerate(encoder.classes_))
print(dictionary)
data.head()


# In[9]:


sns.heatmap(data.corr(),annot=True)


# In[10]:


df=data.copy()
df['Species']=df['Species'].map(dictionary)
sns.pairplot(df, hue='Species',diag_kind='kde')


# In[11]:


data['SepalLengthCm'].hist(color='green',label='SepalLengthCm')
data['SepalWidthCm'].hist(color='yellow',label='SepalWidthCm')
data['PetalLengthCm'].hist(color='blue',label='PetalLengthCm')
data['PetalWidthCm'].hist(color='red',label='PetalWidthCm')

plt.title('Histogram of DataSet Features')
plt.xlabel('Features')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[12]:


print('Range of Features before:')
print(data['SepalLengthCm'].max()-data['SepalLengthCm'].min())
print(data['SepalWidthCm'].max()-data['SepalWidthCm'].min())
print(data['PetalLengthCm'].max()-data['PetalLengthCm'].min())
print(data['PetalWidthCm'].max()-data['PetalWidthCm'].min())


# In[13]:


X=data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
Y=data[['Species']].values.flatten()

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=36)
x_train[:5,],y_train[:5]


# In[14]:


scaler=StandardScaler()
x_train_scale=scaler.fit_transform(x_train,y_train)
x_test_scale=scaler.fit_transform(x_test,y_test)

x_train_scale[:5,],x_test_scale[:5]


# In[15]:


def accuracy_of_model(y_test, y_pred):
    print("Confusion Matrix =>\n{}".format(confusion_matrix(y_test,y_pred)))
    print('Accuracy Score => {}'.format(accuracy_score(y_test, y_pred)))


# In[16]:


from sklearn.linear_model import LogisticRegression

mlogreg=LogisticRegression()
mlogreg.fit(x_train_scale,y_train)
y_train_pred=mlogreg.predict(x_train_scale)
print('Model evaluation for training data: ')
accuracy_of_model(y_train,y_train_pred)

y_test_pred=mlogreg.predict(x_test_scale)
print('\nModel evaluation for test data: ')
accuracy_of_model(y_test,y_test_pred)


# In[17]:


print(dictionary)

fig,ax=plt.subplots(1,4,figsize=(14,4),sharey=True)
col=0
features=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
for triaxis in ax:
    triaxis.scatter(x_test_scale[:,col],y_test,color='blue',label='Actual')
    triaxis.scatter(x_test_scale[:,col],y_test_pred,color='orange',label='Predicted',marker='x')
    triaxis.set_xlabel(features[col])
    triaxis.legend(framealpha=1, frameon=True)
    col+=1
fig.text(0.07,0.5,'SPECIES',va='center',rotation='vertical',fontsize=12)
fig.text(0.35,0.95,'Predicted VS Actual Outputs of Test Data',va='center',rotation='horizontal',fontsize=12)
plt.show(); plt.close()


# In[ ]:




