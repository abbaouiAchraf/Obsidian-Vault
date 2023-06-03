# Exploratory Data Analysis

### **What is Exploratory Data Analysis?**

Exploratory Data Analysis or EDA is used to take insights from the data. Data Scientists and Analysts try to find different patterns, relations, and anomalies in the data using some statistical graphs and other visualization techniques. Following things are part of EDA :
1.  Get maximum insights from a data set
2.  Uncover underlying structure
3.  Extract important variables from the dataset
4.  Detect outliers and anomalies(if any)
5.  Test underlying assumptions
6.  Determine the optimal factor settings

### **Why EDA is important?**

The main purpose of EDA is to detect any errors, outliers as well as to understand different patterns in the data. It allows Analysts to understand the data better before making any assumptions. The outcomes of EDA helps businesses to know their customers, expand their business and take decisions accordingly.

### **How to perform EDA?**

1. Import libraries and load dataset
2. Check for missing values
3. Visualizing the missing values
	``
	```
	sns.heatmap(df.isnull(),cbar=False,cmap='viridis')``
``
4. Replacing the missing values
5. Asking Analytical Questions and Visualizations
	
	This is the most important step in EDA. This step will decide how much can you think as an Analyst. This step varies from person to person in terms of their questioning ability. Try to ask questions related to independent variables and the target variable. For example – how fuel_type will affect the price of the car?

	Let see the heatmap it ll help us to go further :
	```
	plt.figure(figsize=(10,10))
	sns.heatmap(auto.corr(),cbar=True,annot=True,cmap='Blues')
``

![[Pasted image 20230116211924.png]]

# Feature Engineering

## Feature selection

#### 1. Correlation Matrix with Heatmap

Heatmap is a graphical representation of 2D (two-dimensional) data. Each data value represents in a matrix.

Firstly, plot the pair plot between all independent features and dependent features. It will give the relation between dependent and independent features. The relation between the independent feature and the dependent feature is less than 0.2 then choose that independent feature for building a model.

#### 2. Univariate Selection

In this, Statistical tests can be used to select the independent features which have the strongest relationship with the dependent feature. [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) method can be used with a suite of different statistical tests to select a specific number of features.

![Univariate Selection](https://editor.analyticsvidhya.com/uploads/225322.png)

![feature engineering steps univariate selection](https://editor.analyticsvidhya.com/uploads/194213.png)

-   Which feature has the highest score will be more related to the dependent feature and choose those features for the model.

#### 3. ExtraTreesClassifier method

In this method, the ExtraTreesClassifier method will help to give the importance of each independent feature with a dependent feature. Feature importance will give you a score for each feature of your data, the higher the score more important or relevant to the feature towards your output variable.

![ExtraTreesClassifier method](https://editor.analyticsvidhya.com/uploads/780774.png)

![Feature engineering steps ExtraTreesClassifier method output](https://editor.analyticsvidhya.com/uploads/924605.png)

## Handling imbalanced data

Why need to handle imbalanced data? Because of to reduce overfitting and underfitting problem.

_suppose_ a feature has a factor level2(0 and 1). it consists of 1’s is 5% and 0’s is 95%. It is called imbalanced data.

Example:-

![Handling imbalanced data 101](https://editor.analyticsvidhya.com/uploads/7333210.png)

By preventing this problem there are some methods:

#### 1 Under-sampling majority class

Under-sampling the majority class will resample the majority class points in the data to make them equal to the minority class.

![Under-sampling majority class feature engineering steps](https://editor.analyticsvidhya.com/uploads/8541811.png)

#### 2 Over Sampling Minority class by duplication

Oversampling minority class will resample the minority class points in the data to make them equal to the majority class.

![Over Sampling Minority class by duplication feature engineering steps](https://editor.analyticsvidhya.com/uploads/8171312.png)

#### 3 Over Sampling minority class using Synthetic Minority Oversampling Technique (SMOTE)

In this method, synthetic samples are generated for the minority class and equal to the majority class.

![Over Sampling minority class using Synthetic Minority Oversampling Technique (SMOTE)   feature engineering](https://editor.analyticsvidhya.com/uploads/4998913.png)

## Handling outliers

firstly, calculate the skewness of the features and check whether they are positively skewed, negatively skewed, or normally skewed. Another method is to plot the boxplot to features and check if any values are out of bounds or not. if there, they are called outliers.
![Handling outliers fature engineering steps](https://editor.analyticsvidhya.com/uploads/7193414.png)
#### how to handle these outliers: –

first, calculate quantile values at 25% and 75%.
![how to handle these outliers code feature engineering steps](https://editor.analyticsvidhya.com/uploads/8737115.png)
-   next, calculate the Interquartile range

IQR = Q3 – Q1
![IQR = Q3 – Q1](https://editor.analyticsvidhya.com/uploads/1178816.png)
-   next, calculate the upper extreme and lower extreme values

lower extreme=Q1 – 1.5 * IQR
upper extreme=Q3– 1.5 * IQRe
![upper extreme and lower extreme values feature engineering](https://editor.analyticsvidhya.com/uploads/9813417.png)

-   lastly, check the values will lie above the upper extreme or below the lower extreme. if it presents then remove them or replace them with mean, median, or any quantile values.
-   Replace outliers with mean
![Replace outliers with mean feature engineering](https://editor.analyticsvidhya.com/uploads/4137018.png)

-   Replace outliers with quantile values
![Replace outliers with quantile values](https://editor.analyticsvidhya.com/uploads/9596719.png)

-   Drop outliers
![Drop outliers](https://editor.analyticsvidhya.com/uploads/5643320.png)

## Binning

Binning is nothing but any data value within the range is made to fit into the bin. It is important in your data exploration activity. We typically use it to transform continuous variables into discrete ones.

Suppose if we have AGE feature in continuous and we need to divide the age into groups as a feature then it will be useful.

![AGE feature](https://editor.analyticsvidhya.com/uploads/8629321.png)

## Encoding

Why this will apply? because in datasets we may contain object datatypes. for building a model we need to have all features are in integer datatypes. so, Label Encoder and OneHotEncoder are used to convert object datatype to integer datatype.

-   Label Encoding
![Encoding feature engineering](https://editor.analyticsvidhya.com/uploads/9572922.png)

Before applying Label Encoding
![Label Encoding](https://editor.analyticsvidhya.com/uploads/4483023.png)

![Label Encoding feature engineering](https://editor.analyticsvidhya.com/uploads/1144524.png)

After applying label encoding then apply the column transformer method to convert labels to 0 and 1
![label encoding](https://editor.analyticsvidhya.com/uploads/5214325.png)

-   One Hot Encoding:
**One-hot encoding is the representation of categorical variables as binary vectors.** **Label Encoding is converting labels/words into numeric form**.
By applying get_dummies we convert directly categorical to numerical
![One Hot Encoding](https://editor.analyticsvidhya.com/uploads/1468326.png)

## Feature scaling

Why this scaling is applying? because to reduce the variance effect and to overcome the fitting problem. there are two types of scaling methods:

#### 1. Standardization

When this method is used?. when all features are having high values, not 0 and 1.

It is a technique to standardize the independent features that present in a fixed range to bring all values to the same magnitudes.

![Standardization feature engineering steps](https://editor.analyticsvidhya.com/uploads/5627627.png)

In standardization, the mean of the independent features is 0 and the standard deviation is 1.

##### Method 1:

![code](https://editor.analyticsvidhya.com/uploads/1517028.png)

![Method 1 output](https://editor.analyticsvidhya.com/uploads/1602029.png)

##### Method2:

![code Method2](https://editor.analyticsvidhya.com/uploads/6623530.png)

After encoding feature labels are in 0 and 1. This may affect standardization. To overcome this, we use Normalization.

#### 2. Normalisation

Normalization also makes the training process less sensitive by the scale of the features. This results in getting better coefficients after training.

![formula](https://editor.analyticsvidhya.com/uploads/9710531.png)

##### Method 1: -MinMaxScaler

It is a method to rescales the feature to a hard and fast range of [0,1] by subtracting the minimum value of the feature then dividing by the range.

![MinMaxScaler feaure engineering](https://editor.analyticsvidhya.com/uploads/4247132.png)

![](https://editor.analyticsvidhya.com/uploads/6890033.png)

##### Method 2: – Mean Normalization

It is a method to rescales the feature to a hard and fast range of [-1,1] with mean=0.

![Mean Normalization](https://editor.analyticsvidhya.com/uploads/2341734.png)

![Mean Normalization](https://editor.analyticsvidhya.com/uploads/43212Screenshot%20(39).png)

![Mean Normalization feature engineering output](https://editor.analyticsvidhya.com/uploads/7337735.png)





## PCA

# Building a Machine Learning model

Steps to follow while working with your model  : 
- Presentation : Extract and select object features.
- Train Model : Fit the estimator to the data.
- Evaluation.
- Feature and model refinement.

## How to choose the right model
Generally speaking few data and a high number of features would lead us to use algorithm with high bias and low variance so that it generalizes well (Linear Regression, Naïve Bayes, linear SVM, logistic regression). Support Vector Machines are particularly well suited for problem with high numbers of features. As for lots of data and fewer features, this would lead us to use algorithm with low bias and higher variance so that it learns better and doesn’t underfit (KNN, Decision Tree, Random Forest, Kernel SVM, neural nets). Note that you can always use PCA to reduce the number of features. Neural Network have a really hard time learning with few data points, that is one of the reason why machine learning algorithm are sometimes a better options. If your data contains lots of outliers and you don’t want or you can’t get rid of them (you think they are important), you might want to avoid algorithms that are sensible to outliers (linear regression, logistic regression, etc.) Random Forest is on the other hand not sensible to outliers. You can read more about it [here](https://stats.stackexchange.com/questions/187200/how-are-random-forests-not-sensitive-to-outliers).

Some algorithm are made to work better with linear relationship (linear regression, logistic regression, linear SVM). If you data does not contain linear relationships or your input is not numerical or doesn’t have an order (can’t convert into numerical) you might want to try algorithms which can handle high dimensional and complex data structures (Random Forest, Kernel SVM, Gradient Boosting, Neural Nets).

If your target values are binary, logistic regression and SVM are good choice of algorithm. However, if your have a multi-class target, you might need to opt for a more complex model like Random Forest or Gradient Boosting. Also sometimes algorithm have a multi-class equivalent (multiclass logistic regression).

Choosing an algorithm also depends on what is your end goal. Does the model meet the business goals? You might have a threshold in accuracy or other metrics (speed, recall, precision, memory footprint…) that you want or don’t want to surpass. For instance, self-driving cars need really fast prediction times. In that case, you would want to compare the speed of your algorithms and choose accordingly.

Sometimes you might stick to more restrictive algorithms that are easier to train and give a good enough result (Naïve Bayes and Linear and Logistic regression).This might be the case for time restriction, simplicity of the data, interpretability, etc. Approximate methods also naturally tend to avoid overfitting and generalize really well.

Another important thing to consider is the number of parameters you algorithm has. The time required to train a model increases exponentially with the number of parameters, since you have to find the right pattern so it performs well. So if you are time restricted you would want to take this into consideration.

In the case of medical diagnosis, accuracy is much more important than the prediction time and training time. If accuracy is your only goal, you might want to dive into Deep Learning/Neural Nets or use complex model like XGBoost.

Knowing what metric is important in you problem might play a big role in deciding what model to pick. However, metrics are not always the only things that drives your decision.

Model interpretation also plays a role in choosing your model. Sometimes interpretable models are important since they allows us to take concrete action to solve the underlying problem. They is a well known trade-off between accuracy and interpretability, and depending on your end goal you might want to chose the right algorithm.

# BEST PRACTICES :
http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf
