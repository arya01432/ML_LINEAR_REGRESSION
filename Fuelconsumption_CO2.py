#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


df=pd.read_csv('F://FuelConsumptionCo2.csv')
df.head()


# In[51]:


df.drop(['MODELYEAR','MAKE','MODEL','VEHICLECLASS','TRANSMISSION','FUELTYPE'],inplace=True,axis=1)
df.head()


# In[52]:


df.drop(['FUELCONSUMPTION_COMB_MPG','CYLINDERS'],axis=1,inplace=True)
df.head()


# In[53]:


X=df.drop('CO2EMISSIONS',axis=1)
Y=df['CO2EMISSIONS']
print(X)
print(Y)


# In[54]:


plt.figure(figsize=(30,20))
plt.plot(X,Y,'bo')
plt.show()


# # LINEAR REGRESSION USING SCIKIT LEARN 
# ## HERE INDEPENDENT VARIABLE IS FUELCONSUMPTION_CITY

# In[55]:


plt.figure(figsize=(20,10))
plt.plot(X['FUELCONSUMPTION_CITY'],Y,'bo')
plt.show()


# In[56]:


X_city=df['FUELCONSUMPTION_CITY'].values


# In[57]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ## TRAIN OUR ALGO

# In[58]:


X_traincity,X_testcity,Y_traincity,Y_testcity=train_test_split(X_city,Y,test_size=0.2, random_state=4)
X_traincity=X_traincity.reshape(len(X_traincity),1)
log=LinearRegression()
log=log.fit(X_traincity,Y_traincity)
Y_pred_traincity=log.predict(X_traincity)
rmse=np.sqrt(mean_squared_error(Y_traincity,Y_pred_traincity))
print(rmse)
print(log.score(X_traincity,Y_traincity))


# In[59]:


plt.figure(figsize=(20,10))
plt.plot(X_traincity,Y_traincity,'bo')
plt.plot(X_traincity,Y_pred_traincity,'r')
plt.legend()
plt.show()


# ## TESTING OUR ALGO

# In[60]:


X_testcity=X_testcity.reshape(len(X_testcity),1)
Y_testpred=log.predict(X_testcity)
print(np.sqrt(mean_squared_error(Y_testcity,Y_testpred)))
plt.figure(figsize=(20,10))
plt.plot(X_testcity,Y_testcity,'bo',label='points')
plt.plot(X_testcity,Y_testpred,'r',label='regression line')
plt.show()


# ## EQUATION OF ABOVE REGRESSION LINE

# In[61]:


m=log.coef_
c=log.intercept_
print(m,c)
x_city=11.2
y_city=m*x_city+c
print(y_city)


# #  LINEAR REGRESSION USING SCIKIT LEARN
# ## HERE INDEPENDENT VARIABLE IS FUELCONSUMPTION_HWY
# 

# In[62]:


X1=df['FUELCONSUMPTION_HWY'].values


# In[63]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[64]:


X1_trainhwy,X1_testhwy,Y_trainhwy,Y_testhwy=train_test_split(X1,Y,test_size=0.2,random_state=4)
X1_trainhwy=X1_trainhwy.reshape(len(X1_trainhwy),1)
log=LinearRegression()


# In[65]:


log=log.fit(X1_trainhwy,Y_trainhwy)
Y_trainhwypred=log.predict(X1_trainhwy)
print(log.score(X1_trainhwy,Y_trainhwy))


# In[66]:


plt.figure(figsize=(20,10))
plt.plot(X1_trainhwy,Y_trainhwy,'bo')
plt.plot(X1_trainhwy,Y_trainhwypred,'r')
plt.legend()
plt.show()


# ### training error

# In[67]:


rmse=np.sqrt(mean_squared_error(Y_trainhwy,Y_trainhwypred))
print(rmse)
print(log.score(X1_trainhwy,Y_trainhwy))


# In[68]:


X1_testhwy=X1_testhwy.reshape(len(X1_testhwy),1)
Y_testhwypred=log.predict(X1_testhwy)
plt.figure(figsize=(20,10))
plt.plot(X1_testhwy,Y_testhwy,'bo')
plt.plot(X1_testhwy,Y_testhwypred,'r')
plt.legend()
plt.show()


# ### test error

# In[69]:


rmse=np.sqrt(mean_squared_error(Y_testhwy,Y_testhwypred))
print(rmse)
print(log.score(X1_testhwy,Y_testhwy))


# In[70]:


m1=log.coef_
c1=log.intercept_
x1=6.7
y1=m1*x1+c1
print(y1)


# # LINEAR REGRESSION USING SCIKIT LEARN 
# ## HERE INDEPENDENT VARIABLE IS FUELCONSUMPTION_Comb

# In[71]:


X2=df['FUELCONSUMPTION_COMB'].values


# In[72]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[73]:


X2_train_comb,X2_test_comb,Y_train_comb,Y_test_comb=train_test_split(X2,Y,test_size=0.2,random_state=4)
X2_train_comb=X2_train_comb.reshape(len(X2_train_comb),1)
log=LinearRegression()
log=log.fit(X2_train_comb,Y_train_comb)
Y_train_combpred=log.predict(X2_train_comb)
print(log.score(X2_train_comb,Y_train_comb))


# In[74]:


plt.figure(figsize=(20,10))
plt.plot(X2_train_comb,Y_train_comb,'bo')
plt.plot(X2_train_comb,Y_train_combpred,'r')
plt.legend()
plt.show()


# In[75]:


X2_test_comb=X2_test_comb.reshape(len(X2_test_comb),1)
Y_test_comb_pred=log.predict(X2_test_comb)
print(log.score(X2_test_comb,Y_test_comb))


# In[76]:


plt.figure(figsize=(20,10))
plt.plot(X2_test_comb,Y_test_comb,'bo')
plt.plot(X2_test_comb,Y_test_comb_pred,'r')
plt.legend()
plt.show()


# In[77]:


m2=log.coef_
c2=log.intercept_
x2=8.5
y2=m2*x2+c2
print(y2)


# # POLYNOMIAL REGRESSION
# 

# In[128]:


X=df[['ENGINESIZE', 'FUELCONSUMPTION_CITY' ,'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']].values
print(X[:,3:])


# In[133]:



#X=X.reshape(len(X),1)
log=LinearRegression()
log=log.fit(X,Y)
Y_pred=log.predict(X)
print(np.sqrt(mean_squared_error(Y,Y_pred)))
print(log.score(X,Y))


# In[134]:


plt.figure(figsize=(20,10))
y=log.intercept_+log.coef_[0]*X[:,:1]+log.coef_[1]*X[:,1:2]+log.coef_[2]*X[:,2:3]+log.coef_[3]*X[:,3:]
plt.plot(X,Y,'bo')
plt.plot(X,y,'r')
plt.show()


# In[135]:


print(log.coef_)
print(log.intercept_)


# ## equation of polynomial regression

# In[86]:


m1=log.coef_[0]
m2=log.coef_[1]
m3=log.coef_[2]
m4=log.coef_[3]
c=log.intercept_
x1=2
x2=9.9
x3=6.7
x4=8.5

y=m1*x1+m2*x2+m3*x3+m4*x4+c
print(y)


# # POLYNOMIAL REGRESSION WITH ONLY ONE FEATURE
# ## very important to understand

# In[96]:


X_engine=df['ENGINESIZE'].values
X_engine=X_engine.reshape(len(X_engine),1)


# In[97]:


from sklearn.preprocessing import PolynomialFeatures


# In[98]:


poly=PolynomialFeatures(degree=2)
X_engine_poly=poly.fit_transform(X_engine)
X_engine_poly


# In[103]:


log=log.fit(X_engine_poly,Y)
Y_pred=log.predict(X_engine_poly)


# In[104]:


print(log.score(X_engine_poly,Y))


# In[105]:


print(log.coef_)
print(log.intercept_)


# In[109]:


xx=np.arange(0.0,10.0,0.1)
y=log.coef_[1]*xx+log.coef_[2]*np.power(xx,2)+log.intercept_
plt.figure(figsize=(20,10))
plt.plot(X_engine,Y,'bo')
plt.plot(xx,y,'r')
plt.show()


# In[110]:


X_city=df['FUELCONSUMPTION_CITY'].values
X_city=X_city.reshape(len(X_city),1)


# In[111]:


from sklearn.preprocessing import PolynomialFeatures


# In[112]:


poly=PolynomialFeatures(degree=2)
X_city_poly=poly.fit_transform(X_city)
log=log.fit(X_city_poly,Y)
Y_city_pred=log.predict(X_city_poly)
print(log.score(X_city_poly,Y))


# In[114]:


xx=np.arange(4.0,40.0,0.1)
y=log.coef_[1]*xx+log.coef_[2]*np.power(xx,2)+log.intercept_
plt.figure(figsize=(20,10))
plt.plot(X_city,Y,'bo')
plt.plot(xx,y,'r')
plt.show()


# In[ ]:




