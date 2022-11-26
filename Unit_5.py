#!/usr/bin/env python
# coding: utf-8

# In[37]:


# Linear Regression
import matplotlib.pyplot as plt
import statistics as stats
import pandas as pd

data = pd.read_csv("C:/Users/Riya_Asmi/Downloads/height-weight.csv")

X = data['height'].iloc[:30]
y = data['weight'].iloc[:30]


# In[38]:


# Linear Regression
n = len(X)

s_xy = 0 
x_bar = stats.mean(X)
y_bar = stats.mean(y)

x_x = 0
y_y = 0

for i in range(n):
    s_xy += X[i]*y[i]
    x_x += X[i]*X[i]
    y_y += y[i]*y[i]



b_1 = (s_xy - n*x_bar*y_bar)/(x_x - n*x_bar*x_bar)
b_0 = y_bar - b_1*x_bar
r = (s_xy - n*x_bar*y_bar)/((x_x - n*x_bar*x_bar)*(y_y - n*y_bar*y_bar))**0.5


print(f"Beta 1 is {b_1}")
print(f"Beta 0 is {b_0}")
print(f"r is {r}")


# In[39]:


# Linear Regression
plt.scatter(X,y)
plt.xlabel('Height')
plt.ylabel('Weight')

Y_hat = []
for i in range(n):
    Y_hat.append(b_0 + b_1*X[i])
plt.plot(X,Y_hat)

plt.show()


# In[40]:


# Linear Regression
x_inp = int(input("Enter height: "))
y_pred = b_0 + b_1*x_inp
print(f"Predicted weight for height {x_inp} is {y_pred}")


# In[41]:


#Creating a Residual Plot
residual = []
for i in range(n):
    residual.append(y[i] - Y_hat[i])
plt.scatter(X,residual)
plt.title('Residual plot')
plt.xlabel('Height')
plt.ylabel('Residual')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[31]:


#Pearson Correlation Coefficeint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

data={
    "x":[1,4,5,2,6,10,9,8,7,3],
    "y":[1,2,3,4,5,6,7,8,9,10]
}

df=pd.DataFrame(data)
df


# In[32]:


#Pearson Correlation Coefficeint
x_mean=df["x"].mean()
y_mean=df["y"].mean()

s=0
p=0

for i in range (len(data["x"])):
    s+=(data["x"][i]-x_mean)*(data["y"][i]-y_mean)
    p+=(data["x"][i]-x_mean)*(data["x"][i]-x_mean)
b1=s/p

b0=y_mean-b1*x_mean
    
x_hat=np.linspace(0,20,200)
y_hat=b1*x_hat+b0
plt.plot(x_hat,y_hat)
plt.scatter(df["x"],df["y"])
plt.show()
print("y_hat=",b1,"x_hat +",b0,sep="")

def f(x):
    return b1*x+b0

f(8)


a=0
b=0
c=0
for i in range(len(data["x"])):
    a+=((data["x"][i]-x_mean)*(data["y"][i]-y_mean))
    b+=(data["x"][i]-x_mean)*(data["x"][i]-x_mean)
    c+=(data["y"][i]-y_mean)*(data["y"][i]-y_mean)
b=math.sqrt(b)    
c=math.sqrt(c)
r=a/(b*c)

print("Pearson Corelation Coefficient value is ",r)


# In[25]:


#Power Of Test
from math import sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

#H0 : mu >= 50
#H1 : mu < 50

mu0 = 50
n = 30
sigma = 10
alpha = 0.05
n_list = [20,45,280,72,140]
alpha_list = [0.01,0.09,0.5,0.01,0.06]
sigma_list = [6,13,69,40,90]


# In[26]:


#Power of Test
# Moving in the direction of the alternative hypothesis: Power increases
z_nd = norm.ppf(0.05)
se = sigma/np.sqrt(n)

critical_x_bar = mu0 + z_nd*se
power = []
for i in range(40,mu0):
    z_ad = (critical_x_bar - i)/se
    power.append(norm.cdf(z_ad))

plt.plot(range(40,mu0),power)
plt.xlabel('Mean')
plt.ylabel('Power')
plt.show()


# In[27]:


#Power of Test
for sample_size in n_list:
    z_nd = norm.ppf(0.05)
    se = sigma/np.sqrt(sample_size)
    critical_x_bar = mu0 + z_nd*se
    power = []
    for i in range(40,mu0):
        z_ad = (critical_x_bar - i)/se
        power.append(norm.cdf(z_ad))
    plt.plot(range(40,mu0),power,label = 'Sample Size : '+str(sample_size))
plt.legend()
plt.show()


# In[28]:


#Power of Test
for sample_size in alpha_list:
    z_nd = norm.ppf(0.05)
    se = sigma/np.sqrt(sample_size)
    critical_x_bar = mu0 + z_nd*se
    power = []
    for i in range(40,mu0):
        z_ad = (critical_x_bar - i)/se
        power.append(norm.cdf(z_ad))
    plt.plot(range(40,mu0),power,label = 'Alpha values: '+str(sample_size))
plt.legend()
plt.show()


# In[29]:


#Power of Test
for sample_size in sigma_list:
    z_nd = norm.ppf(0.05)
    se = sigma/np.sqrt(sample_size)
    critical_x_bar = mu0 + z_nd*se
    power = []
    for i in range(40,mu0):
        z_ad = (critical_x_bar - i)/se
        power.append(norm.cdf(z_ad))
    plt.plot(range(40,mu0),power,label = 'Sigma values : '+str(sample_size))
plt.legend()
plt.show()


# In[ ]:




