# cheat_sheet_Algorithm

Linear Regression
-----------------------------------------------------------
Assumptions:
1)There should be a linear and additive relationship between dependent (response) variable and independent (predictor) variable(s).
2)data should have little to no multicollinearity.
3)Residuals should have homoscedasticity
4)Residuals should be normally distributed.

Cost function:
MSE formula = (1/n) * Σ(x – (mx+c))^2

------------------------------------------------------------

1)Import other necessary libraries like pandas, numpy...

2)from sklearn import linear_model
2.1)import dataset

3)Load Train and Test datasets

4)Identify feature and response variable(s) 
# value ranges between -∞ to ∞ used to predict contionous variable. 

5)values must be numeric and numpy arrays 
x_train=input_variables_values_training_datasets 
y_train-target_variables_values_training_datasets 
x_test=input_variables_values_test_datasets 

6)Create linear regression object
linear linear_model.LinearRegression()

7)Train the model using the training sets and check score
linear.fit(x_train, y_train) linear.score(x_train, y_train) 

8)prnt Equation coefficient and Intercept 
print('Coefficient: \n', linear.coef_) 
print("Intercept: \n', linear.intercept_) 

9) predict the output
predicted linear.predict(x_test)

================================================================
Logistic Regression

why MSE not used in Logistic Regression?
To evaluate the performance of the model, we calculate the loss. 
The most commonly used loss function is the mean squared error. 
But in logistic regression, as the output is a probability value between 0 or 1, mean squared error wouldn't be the right choice.
So, instead, we use the cross-entropy loss function.
-----------------------
NOTE:it predict the relationship between predictors (our independent variables) 
and a predicted variable (the dependent variable) where the dependent variable is binary

cost function:sigmoid function=1/1+e^(-y)ie:y=mx+c

--------------------------------------------------------------------
1)Import Library
from sklearn.linear_model import LogisticRegression
1.1)Import dataset

2)Assume you have, X (predictor) and Y (target) #for training data set and x_test(predictor)
3)import  test_dataset

4)Create logistic regression object

model LogisticRegression()

5)Train the model using the training sets and check score

model.fit(x, y)

model.score (x, y)

6)Equation coefficient and Intercept 
print('Coefficient: \n', model.coef_) print('Intercept: \n', model.intercept_) 

7)predict output
predicted model.predict(x_test)

8)print accuracy score
accuracy_score(data['y'],pred)


===================================================================
KNN 
NOTE:i)input different k values
ii) optimal K value will the point in elbow curve where the curve bends.
iii)2 types of scaling : min-max and standard scaling.
-------------------------------------------------------------
1)Import Library
from sklearn.neighbors import KNeighbors Classifier
1.1)import dataset
2)Assume you have, X (predictor) and Y (target) for 
3)training data set and x_test (predictor) of test_dataset 
4)Create KNeighbors classifier object model
KNeighbors Classifier (n_neighbors=6)
5)default value for n_neighbors is 5
6)Train the model using the training sets and check score
model.fit(x, y)
7)Predict Output
predicted model.predict(x_test)
=====================================================================
