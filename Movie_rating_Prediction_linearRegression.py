import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
data=pd.read_excel("C:\\Users\\pints\\Desktop\\python pro\\DATASET\\cwebsite-ratings-data.xlsx")
print(data.head())

#============plotiing==============

plt.figure(figsize=(12,6))
plt.scatter(data['User'],data['ratings'],c='red')
plt.xlabel("USERS")
plt.ylabel("Rating(out of 5)")
plt.show()


#now creating linear approximation
X=data['User'].values.reshape(-1,1)
Y=data['ratings'].values.reshape(-1,1)
reg=LinearRegression()
reg.fit(X,Y)

#reg.coef_calculate slope,reg.intercept_calculate 'c'

print("The linear model is: Y={:.5}X+{:.5}".format(reg.coef_[0][0],reg.intercept_[0]))

#now creating prediction

predictions=reg.predict(X)# calc new Y
plt.figure(figsize=(12,6))
plt.scatter(data['User'],data['ratings'],c='black')
plt.xlabel("USERS")
plt.ylabel("Rating")


plt.plot(data['User'],predictions,c='red',linewidth=2)
plt.xlabel("USERS")
plt.ylabel("Rating")
plt.show()

#now assesing efficiancy R-Squared model

X=data['User']
Y=data['ratings']
X2=sm.add_constant(X)


est=sm.OLS(Y,X2)
est2=est.fit()
print(est2.summary())

print("Enter user number")
p=int(input())
rate=p*reg.coef_[0][0]+reg.intercept_[0]
print("User will rate:",rate)





