#!/usr/bin/env python
# coding: utf-8

# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20.0,10.0)


# In[36]:
#read datasets

data=pd.read_csv('F:\\headbrain.csv')
print(data.shape)
data.head(10) #print 10 instances of dataset from top


# In[37]:


data.tail(5) #print 5 instances of dataset from bottom


# ## assign input and output in X,Y respectively

# In[19]:


X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values


# ## for finding mean of input and output variables

# In[20]:


mean_x=np.mean(X)
mean_y=np.mean(Y)
print(mean_x)
print(mean_y)


# ## for finding slope and intercept of line m-slope and c-intercept

# In[21]:


l=len(X)
sums=0
sq_sum=0
for i in range(l):
    sums+=(X[i]-mean_x)*(Y[i]-mean_y)
    sq_sum+=(X[i]-mean_x)**2
m=sums/sq_sum
print(m)
c=mean_y-(m*mean_x)
print(c)
print(sums,sq_sum)


# ## Ploting values and regression line

# In[23]:


max_x=np.max(X)+100
min_x=np.min(X)-100

x=np.linspace(max_x,min_x,1000)
y = m*x + c

plt.plot(x,y,color='#58b970',label='Regression Line')
plt.scatter(X,Y,color='#ef5423',label='Scatter Plot')

plt.xlabel('Head size in cm3')
plt.ylabel('Brain weight in grams')
plt.legend()
plt.show()


# ## Finding RMSE - root mean square error

# In[30]:


error_sum=0
for i in range(l):
    error_sum+=(Y[i]-(m * X[i]+c))**2

avg_error_sum=error_sum/l
print((avg_error_sum)**.5)


# ## finding R^2 such that R^2 is less than 1 and not negative

# In[31]:


ssr=error_sum
sst=0
for i in range(l):
    sst+=(Y[i]-mean_y)**2
    
r2=1-(ssr/sst)
print(r2)

