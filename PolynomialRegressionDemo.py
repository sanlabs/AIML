
# Importing the libraries 
import numpy as np #used for Reshape Array
import matplotlib.pyplot as plt # used for Plotting graph
import pandas as pd #used for reading csv
#sklearn used for Regression Models(linear & Polynomial)

print("This line will be printed....Lets start")

#Define Input Arrays
X_1DArray = []
Y_1DArray = []

#Step1: Importing the dataset 
csv_data = pd.read_csv('Linear-Polynomial-Data.csv') 
 
#Showing column numbers and Row count
print(csv_data.shape) 
#csv_data.head() #will print number of rows and cols in csv

#Step 2: Dividing the dataset into 2 Arrays
X_1DArray = csv_data['X'].values
Y_1DArray = csv_data['Y'].values

print("X_1DArray" + str(X_1DArray))
print("Y_1DArray" + str(Y_1DArray))

#Step 3: Fitting dataset tpo Linear Regression
#Reshape 1Darray to 2Darray
X_2DArray = []
Y_2DArray = []
X_2DArray = np.reshape(X_1DArray, (len(X_1DArray),1)) # converting to matrix of n X 1
Y_2DArray = np.reshape(Y_1DArray, (len(Y_1DArray),1)) # converting to matrix of n X 1

print("X_2DArray" + str(X_2DArray))
print("Y_2DArray" + str(Y_2DArray))


from sklearn.linear_model import LinearRegression 
line1 = LinearRegression() 
line1.fit(X_2DArray, Y_2DArray) 

# Step: Visualising the Linear Regression results 
plt.scatter(X_1DArray, Y_1DArray, color = 'blue') 
plt.plot(X_2DArray, line1.predict(X_2DArray), color = 'blue') 
plt.title('Linear Regression') 
plt.xlabel('X axis') 
plt.ylabel('Y axis') 

plt.show() 

## Step: Fitting dataset to Polynomial Regression Degree 2
from sklearn.preprocessing import PolynomialFeatures 
poly2D = PolynomialFeatures(degree = 2) 
X_poly = poly2D.fit_transform(X_2DArray) 
poly2D.fit(X_poly, Y_2DArray) 
line2 = LinearRegression() 
line2.fit(X_poly, Y_2DArray) 


# Visualising the Polynomial Degree-2 results 
plt.scatter(X_1DArray, Y_1DArray, color = 'blue') 
plt.plot(X_2DArray, line1.predict(X_2DArray), color = 'blue') 
plt.plot(X_2DArray, line2.predict(poly2D.fit_transform(X_2DArray)), color = 'red') 
plt.title('Polynomial Regression Degree-2') 
plt.xlabel('X axis') 
plt.ylabel('Y axis') 

plt.show() 


#Step:Fitting dataset to Polynomial Regression Degree-3
from sklearn.preprocessing import PolynomialFeatures 
poly3d = PolynomialFeatures(degree = 3) 
X_poly = poly3d.fit_transform(X_2DArray) 
poly3d.fit(X_poly, Y_2DArray) 
line3 = LinearRegression() 
line3.fit(X_poly, Y_2DArray) 

#Visualising the Linear, Polynomial Degree-2 & Polynomial Degree-3
plt.scatter(X_1DArray, Y_1DArray, color = 'blue') 
plt.plot(X_2DArray, line1.predict(X_2DArray), color = 'blue') 
plt.plot(X_2DArray, line2.predict(poly2D.fit_transform(X_2DArray)), color = 'red') 
plt.plot(X_2DArray, line3.predict(poly3d.fit_transform(X_2DArray)), color = 'green') 
plt.title('Polynomial Regression Degree-3') 
plt.xlabel('X axis') 
plt.ylabel('Y axis') 

plt.show() 



