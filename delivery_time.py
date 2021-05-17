'''
Delivery_time -> Predict delivery time using sorting time

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

# Load delivery_time.csv as pandas dataframe
delivery_time = pd.read_csv("delivery_time.csv")

# View Data
print(delivery_time.head())

# Perform EDA
# Identifying the number of features or columns
print("Dataset has {} columns".format(len(delivery_time.columns)))

# Identifying the features or columns
print("The columns in our dataset are :",list(delivery_time.columns))

# We can use shape to see the size of the dataset
print(delivery_time.shape) # has 21 rows and 2 columns

# Identifying the data types of features
print(delivery_time.dtypes)

# Checking if the dataset has empty cells 
print(delivery_time.isnull().values.any()) # Returns False as there are no null values

# Identifying the number of empty cells by features or columns
print(delivery_time.isnull().sum())

# Info can also be used to check the datatypes 
delivery_time.info() # shows the datatypes, non-null count of all the columns


# Describe will help to see how numerical data has been spread. 
# We can see some of the measure of central tendency, and percentile values.
print(delivery_time.describe())

# Graphical Univariate Analysis:
# Histogram

# delivery_time[['Delivery Time', 'Sorting Time']].boxplot() # .hist()
# plt.title("Boxplot of numerical features")
# plt.show()

plt.subplot(2,4,1)
plt.hist(delivery_time['Delivery Time'], density=False)
plt.title("Histogram of 'Delivery Time'")
plt.subplot(2,4,5)
plt.hist(delivery_time['Sorting Time'], density=False)
plt.title("Histogram of 'Sorting Time'")

plt.subplot(2,4,2)
sns.distplot(delivery_time['Delivery Time'], kde=True)
plt.title("Density distribution of 'Delivery Time'")
plt.subplot(2,4,6)
sns.distplot(delivery_time['Sorting Time'], kde=True)
plt.title("Density distribution of 'Sorting Time'")
# boxplot
plt.subplot(2,4,3)
# plt.boxplot(delivery_time['Delivery Time'])
sns.violinplot(delivery_time['Delivery Time'])
# plt.title("Boxlpot of 'Delivery Time'")
plt.title("Violin plot of 'Delivery Time'")
plt.subplot(2,4,7)
# plt.boxplot(delivery_time['Sorting Time'])
sns.violinplot(delivery_time['Sorting Time'])
# plt.title("Boxlpot of 'Sorting Time'")
plt.title("Violin plot of 'Sorting Time'")

# Normal Q-Q plot
plt.subplot(2,4,4)
probplot(delivery_time['Delivery Time'], plot=plt)
plt.title("Q-Q plot of Delivery Time")
plt.subplot(2,4,8)
probplot(delivery_time['Sorting Time'], plot=plt)
plt.title("Q-Q plot of Sorting Time")
# plt.show()

# The data points look normally distributed from the graphs. Lets run shapiro test and confirm 
pValueDel = stats.shapiro(delivery_time['Delivery Time'])[1]
print("P Value from shapiro test of 'Delivery Time' = {}".format(pValueDel)) # p > 0.05. We fail to reject Null Hypothesis. Data is normally distributed

pValueSort = stats.shapiro(delivery_time['Sorting Time'])[1]
print("P Value from shapiro test of 'Sorting Time' = {}".format(pValueSort)) # p > 0.05. We fail to reject Null Hypothesis. Data is normally distributed

# Bivariate visualization
# Scatterplot & Line plots
# plt.scatter(delivery_time['Delivery Time'], delivery_time['Sorting Time'], edgecolors='face', alpha=0.5, c='#17becf')
plt.subplot(1,3,1)
sns.scatterplot(data=delivery_time, x="Sorting Time", y="Delivery Time", hue="Delivery Time", alpha=0.5)
plt.title("Scatter plot")
plt.subplot(1,3,2)
sns.lineplot(data=delivery_time, x="Sorting Time", y="Delivery Time")
plt.title("Line plot of Delivery Time, Sorting Time")
plt.subplot(1,3,3)
sns.lineplot(data=delivery_time)
plt.title('Line Plot')
# plt.show()

# heatmap
plt.subplot(1, 2, 1)
sns.heatmap(data=delivery_time, cmap="YlGnBu", annot = True)
plt.title("Heatmap using seaborn")
plt.subplot(1, 2, 2)
plt.imshow(delivery_time, cmap ="YlGnBu")
plt.title("Heatmap using matplotlib")
# plt.show()

# Joint plot
sns.jointplot(x = "Sorting Time", y = "Delivery Time", kind = "reg", data = delivery_time)
plt.title("Joint plot using sns")
# kind can be hex, kde, scatter, reg, hist. When kind='reg' it shows the best fit line.
# plt.show()

# Check if there is any correlation between the variables
print("Correlation: "+ '\n', delivery_time.corr()) # 0.825 which is moderate positive correlation
# Draw a heatmap for correlation matrix
plt.subplot(1,1,1)
sns.heatmap(delivery_time.corr(), annot=True)
# plt.show()


# Standardisation
# std_scaler = StandardScaler()
# delivery_time['Std_delivery_time'] = std_scaler.fit_transform(delivery_time[['Delivery Time']])
# delivery_time['Std_sorting_time'] = std_scaler.fit_transform(delivery_time[['Sorting Time']])

delivery_time['Std_delivery_time'] = preprocessing.scale(delivery_time[['Delivery Time']])
delivery_time['Std_sorting_time'] = preprocessing.scale(delivery_time[['Sorting Time']])

# Normalization
delivery_time['Norm_delivery_time'] = preprocessing.normalize(delivery_time[['Delivery Time']], axis=0)
delivery_time['Norm_sorting_time'] = preprocessing.normalize(delivery_time[['Sorting Time']], axis=0)
# print(delivery_time.iloc[:, -3:])



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
	# Checking the MSE

	# Predicting the output
	y_pred = regressor.predict(x)

	print("Mean squared error(MSE): %.2f"
	      % mean_squared_error(y, y_pred))
	# Checking the R2 value
	print("Coefficient of determination: %.3f"
	      % r2_score(y, y_pred)) # Evaluated the performance of the model # says much percentage of data points are falling on the best fit line

# Driver code
# regression(delivery_time[['Delivery Time', 'Sorting Time']]) # 0.682 accuracy
# regression(delivery_time[['Std_delivery_time', 'Std_sorting_time']]) # 0.682 accuracy
# regression(delivery_time[['Norm_delivery_time', 'Norm_sorting_time']]) # 0.682 accuracy




### Regression using statsmodels
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

# OLS_model(delivery_time[['Delivery Time', 'Sorting Time']]) # 0.682 accuracy
# OLS_model(delivery_time[['Std_delivery_time', 'Std_sorting_time']]) # 0.682 accuracy
# OLS_model(delivery_time[['Norm_delivery_time', 'Norm_sorting_time']]) # 0.682 accuracy


# R-Squared is the coef of determination. Value closer to 1 means model is good.
# Adjusted R-squared value and R-squared values must be close, if not, it means we added an irrelevant feature in model prediction
# F-statistic 
# 




# Let us create a new dataset and apply transformations
df = delivery_time.loc[:, ('Delivery Time', 'Sorting Time')]
# print(df.head())


# Transforming variables for accuracy

# sqrt transformation on the input feature
df['sqrt_sorting_time'] = np.sqrt(delivery_time['Sorting Time'])
# Log transformation on the input feature
df['log_sorting_time'] = np.log1p(delivery_time['Sorting Time'])

# Exponential transformation (log transformation on the input feature)
df['log_delivery_time'] = np.log1p(delivery_time['Delivery Time'])

# Exponential transformation on the input
df['exp_sorting_time'] = np.exp(delivery_time['Sorting Time'])

# Other transformations
df['1/sqrt_delivery_time'] = (1/np.sqrt(delivery_time['Delivery Time']))
df['log_2_sorting_time'] = np.log1p(delivery_time['Sorting Time']*2) 

# Polynomial
df['sq_sorting_time'] = df['Sorting Time']*df['Sorting Time']

print(df.iloc[:, -4:].head())

# OLS_model(df[['Delivery Time', 'sqrt_sorting_time']]) # 0.696 accuracy
# OLS_model(df[['Delivery Time', 'log_sorting_time']]) # 0.697 accuracy
# OLS_model(df[['log_delivery_time', 'Sorting Time']]) # 0.711 accuracy
# OLS_model(df[['Delivery Time', 'exp_sorting_time']]) # 0.361 accuracy
# OLS_model(df[['log_delivery_time', 'sqrt_sorting_time']]) # 0.747 accuracy
# OLS_model(df[['1/sqrt_delivery_time', 'log_sorting_time']]) # 0.776 accuracy
OLS_model(df[['1/sqrt_delivery_time', 'log_2_sorting_time']]) #0.782 accuracy

# Polynomial regression
df['deliveryTime'] = df['Delivery Time']
df['sortingTime'] = df['Sorting Time']
print(df.columns)
model_quad = smf.ols('deliveryTime~sortingTime+sq_sorting_time', data=df).fit()
# print(model_quad.summary()) # 0.693 accuracy


# Regression using statsmodels.formula.api
model = smf.ols('deliveryTime~sortingTime', data=df).fit()
print(model_quad.summary())
y_pred = model.predict(df['sortingTime'])
print(y_pred)
plt.figure(figsize=(10, 7))
plt.scatter(df['sortingTime'], df['deliveryTime'], color='teal')
plt.plot(df['sortingTime'], y_pred, color='red')
plt.title('Simple Linear Regression')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()

# By applying 1/sqrt(output), log(2*input) we are able to increase the accuracy from 68.2% to 78.2%