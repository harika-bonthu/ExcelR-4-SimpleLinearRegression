'''
Salary_hike -> Build a prediction model for Salary_hike

------------------------------------------------------------

Build a simple linear regression model by performing EDA and do necessary transformations and
select the best model using R or Python.
'''
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import probplot
import statsmodels.api as sm # We need to add constant using .add_constant()
import statsmodels.formula.api as smf # a constant is automatically added to the data and an intercept in fitted
# from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Salary_Data.csv as pandas dataframe
Salary_Data = pd.read_csv("Salary_Data.csv")

# View Data
print(Salary_Data.head())

# Perform EDA
# Identifying the number of features or columns
print("Dataset has {} columns".format(len(Salary_Data.columns)))

# Identifying the features or columns
print("The columns in our dataset are :",list(Salary_Data.columns))

# We can use shape to see the size of the dataset
print(Salary_Data.shape) # has 30 rows and 2 columns

# Identifying the data types of features
print(Salary_Data.dtypes)

# Checking if the dataset has empty cells 
print(Salary_Data.isnull().values.any()) # Returns False as there are no null values

# Identifying the number of empty cells by features or columns
print(Salary_Data.isnull().sum())

# Info can also be used to check the datatypes 
Salary_Data.info() # shows the datatypes, non-null count of all the columns


# Describe will help to see how numerical data has been spread. 
# We can see some of the measure of central tendency, and percentile values.
print(Salary_Data.describe())

# Graphical Univariate Analysis:
# Histogram

# Salary_Data[['YearsExperience', 'Salary']].boxplot() # .hist()
# plt.title("Boxplot of all numerical features")
# plt.show()

plt.subplot(2,4,1)
plt.hist(Salary_Data['YearsExperience'], density=False)
plt.title("Histogram of 'YearsExperience'")
plt.subplot(2,4,5)
plt.hist(Salary_Data['Salary'], density=False)
plt.title("Histogram of 'Salary'")

plt.subplot(2,4,2)
sns.distplot(Salary_Data['YearsExperience'], kde=True)
plt.title("Density distribution of 'YearsExperience'")
plt.subplot(2,4,6)
sns.distplot(Salary_Data['Salary'], kde=True)
plt.title("Density distribution of 'Salary'")
# boxplot
plt.subplot(2,4,3)
# plt.boxplot(Salary_Data['YearsExperience'])
sns.violinplot(Salary_Data['YearsExperience'])
# plt.title("Boxlpot of 'YearsExperience'")
plt.title("Violin plot of 'YearsExperience'")
plt.subplot(2,4,7)
# plt.boxplot(Salary_Data['Salary'])
sns.violinplot(Salary_Data['Salary'])
# plt.title("Boxlpot of 'Salary'")
plt.title("Violin plot of 'Salary'")

# Normal Q-Q plot
plt.subplot(2,4,4)
probplot(Salary_Data['YearsExperience'], plot=plt)
plt.title("Q-Q plot of 'YearsExperience'")
plt.subplot(2,4,8)
probplot(Salary_Data['Salary'], plot=plt)
plt.title("Q-Q plot of 'Salary'")
# plt.show()

# From the graphs, YearsExperience looks normal, Salary doesn't look normal. Lets check it using the Shapiro test
pValueExp = stats.shapiro(Salary_Data['YearsExperience'])[1]
print("P Value from shapiro test of 'YearsExperience' = {}".format(pValueExp)) # p > 0.05. We fail to reject Null Hypothesis. Data is normally distributed

pValueSal = stats.shapiro(Salary_Data['Salary'])[1]
print("P Value from shapiro test of 'Salary' = {}".format(pValueSal)) # p < 0.05. We reject Null Hypothesis. Data is not normally distributed

# Bivariate visualization
# Scatterplot & Line plots
# plt.scatter(Salary_Data['YearsExperience'], Salary_Data['Salary'], edgecolors='face', alpha=0.5, c='#17becf')
plt.subplot(1,3,1)
sns.scatterplot(data=Salary_Data, x="YearsExperience", y="Salary", hue="YearsExperience", alpha=0.5)
plt.title("Scatter plot")
plt.subplot(1,3,2)
sns.lineplot(data=Salary_Data, x="YearsExperience", y="Salary")
plt.title("Line plot of YearsExperience, Salary")
plt.subplot(1,3,3)
sns.lineplot(data=Salary_Data)
plt.title('Line Plot')
# plt.show()

# heatmap
plt.subplot(1, 2, 1)
sns.heatmap(data=Salary_Data, cmap="YlGnBu", annot = True)
plt.title("Heatmap using seaborn")
plt.subplot(1, 2, 2)
plt.imshow(Salary_Data, cmap ="YlGnBu")
plt.title("Heatmap using matplotlib")
# plt.show()

# Joint plot
sns.jointplot(x = "YearsExperience", y = "Salary", kind = "reg", data = Salary_Data)
plt.title("Joint plot using sns")
# kind can be hex, kde, scatter, reg, hist. When kind='reg' it shows the best fit line.
# plt.show()

# Check if there is any correlation between the variables
print("Correlation: "+ '\n', Salary_Data.corr()) # 0.98 which is high positive correlation
# Draw a heatmap for correlation matrix
plt.subplot(1,1,1)
sns.heatmap(Salary_Data.corr(), annot=True)
# plt.show()


# Standardisation
# std_scaler = StandardScaler()
# Salary_Data['Std_YearsExp'] = std_scaler.fit_transform(Salary_Data[['YearsExperience']])
# Salary_Data['Std_Salary'] = std_scaler.fit_transform(Salary_Data[['Salary']])

Salary_Data['Std_YearsExp'] = preprocessing.scale(Salary_Data[['YearsExperience']])
Salary_Data['Std_Salary'] = preprocessing.scale(Salary_Data[['Salary']])

# Normalization
Salary_Data['Norm_YearsExp'] = preprocessing.normalize(Salary_Data[['YearsExperience']], axis=0)
Salary_Data['Norm_Salary'] = preprocessing.normalize(Salary_Data[['Salary']], axis=0)
# print(Salary_Data.iloc[:, -3:])



### Regression using sklearn library

def regression(df):
	# defining the independent and dependent features
	x = df.iloc[:, 1:2]
	y = df.iloc[:, 0:1] 
	# print(x,y)

	# Instantiating the LinearRegression object
	regressor = LinearRegression()

	# Training the model
	regressor.fit(x,y)

	# Checking the coefficients for the prediction of each of the predictor
	print("Coefficients of the predictor: ",regressor.coef_)
	# Checking the intercept
	print("Intercept: ",regressor.intercept_)

	# Predicting the output
	y_pred = regressor.predict(x)

	# Checking the MSE
	print("Mean squared error(MSE): %.2f"
	      % mean_squared_error(y, y_pred))
	# Checking the R2 value
	print("Coefficient of determination: %.3f"
	      % r2_score(y, y_pred)) # Evaluates the performance of the model # says much percentage of data points are falling on the best fit line

	# visualizing the results.
	plt.figure(figsize=(10, 7))
	# Scatter plot of input and output values
	plt.scatter(x,y, color='teal')
	# plot of the input and predicted output values
	plt.plot(x, y_pred, color='Red', linewidth=2)
	plt.title('Simple Linear Regression')
	plt.xlabel('YearExperience')
	plt.ylabel('Salary')
	plt.show() 

# Driver code
regression(Salary_Data[['Salary', 'YearsExperience']]) # 0.957 accuracy
# regression(Salary_Data[['Std_Salary', 'Std_YearsExp']]) # 0.957 accuracy
regression(Salary_Data[['Norm_Salary', 'Norm_YearsExp']]) # 0.957 accuracy




### Regression using statsmodels.api
def OLS_model(df):
	# Add constant

	# defining the independent and dependent features
	x = df.iloc[:, 1:2]
	y = df.iloc[:, 0:1] 
	x = sm.add_constant(x)
	# print(x)
	model = sm.OLS(y, x)
	results = model.fit()
	# print('\n'+"Confidence interval:"+'\n', results.conf_int(alpha=0.05, cols=None)) #Returns the confidence interval of the fitted parameters. The default alpha=0.05 returns a 95% confidence interval.
	print('\n'"Model parameters:"+'\n',results.params)
	print(results.summary())
	# y_pred = model.predict(x)
	# print(y_pred)
OLS_model(Salary_Data[['Salary', 'YearsExperience']]) # 0.957 accuracy
# OLS_model(Salary_Data[['Std_Salary', 'Std_YearsExp']]) # 0.957 accuracy
OLS_model(Salary_Data[['Norm_Salary', 'Norm_YearsExp']]) # 0.957 accuracy



# Regression using statsmodels.formula.api
def smf_ols(df):
    # defining the independent and dependent features
    x = df.iloc[:, 1:2]
    y = df.iloc[:, 0:1] 
	# print(x)
    # train the model
    model = smf.ols('y~x', data=df).fit()
    # print model summary
    print(model.summary())
    
    # Predict y
    y_pred = model.predict(x)
	# print(type(y), type(y_pred))
	# print(y, y_pred)

    y_lst = y.Salary.values.tolist()
	# y_lst = y.iloc[:, -1:].values.tolist()
    y_pred_lst = y_pred.tolist()
    
	# print(y_lst)
        
    data = [y_lst, y_pred_lst]
	# print(data)
    res = pd.DataFrame({'Actuals':data[0], 'Predicted':data[1]})
	# print(res)
    
    plt.scatter(x=res['Actuals'], y=res['Predicted'])
    plt.ylabel('Predicted')
    plt.xlabel('Actuals')
    plt.show()
    
    res.plot(kind='bar',figsize=(10,6))
    plt.show()

# Driver code
smf_ols(Salary_Data[['Salary', 'YearsExperience']]) # 0.957 accuracy
# smf_ols(Salary_Data[['Norm_Salary', 'Norm_YearsExp']]) # 0.957 accuracy
                            

#### We achieved a total accuracy of 95.7% which is pretty good.                            