# Machine-Learning-A-Z
## Data Preprocessing
###### Spliting Data
   In this section we import the data, clean it and we split the data in to training and test set. Usually about 20% of the data is used for test set and 80% for training set. We'll use our training set to build the model. Then we will take the data from the test set and apply our model to them. So they(test data) haven't been part of the model creation process.The model has no information about test data. And the model predicting certain values. But the good news is that because this is something we separated in advance as part of the data that was given to us we actually know the actual values. So now we can compare the predicted values which were generated using a model that has never seen test data before to the actual values. And so from that we can evaluate our model.
   
   ###### Splitting the dataset
      from sklearn.model_selection import train_test_split
      x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
   
   ###### Feature Scaling
   Now, even without knowing anything about feature scaling, please remember that feature scaling is always applied to columns. Feature scaling is never applied across columns, so you wouldn't apply feature scaling to data inside a row.let's have a look at what feature scaling actually is. So there are multiple types of feature scaling, multiple techniques. We're going to look at the two main ones, **normalization** and **standardization**.
   
   **Normalization** is the process of taking the minimum inside a column, subtracting that minimum from every single value inside that column, and then dividing by the difference between the maximum and the minimum. So basically, every single value in a column is adjusted this way and you will end up with a new column or an adjusted column with values which are all between **0 and 1**.
   ![Noramilization formula](https://cdn.wallstreetmojo.com/wp-content/uploads/2019/05/Normalization-Formula.jpg)
   
   **Standardization** Standardization, on the other hand, the process is similar, but instead of subtracting the minimum, we subtract the average, and we divide by the standard deviation. As a result, all of the values, or almost all of the values inside the column will be between **-3 and 3**.
   
   ![Standardization formula](https://miro.medium.com/v2/resize:fit:970/0*3E-1O6yCamLFE3qE)
   
   ![Standard deviation](https://i.ytimg.com/vi/Uk98hiMQgN0/maxresdefault.jpg)
   
   ###### Feature Scaling Code
      from sklearn.preprocessing import StandardScaler
      sc=StandardScaler()
      x_train=sc.fit_transform(x_train)
      x_test=sc.transform(x_test)


   point1: We do not apply fit method on test set because indeed the features of the test set need to be scaled by the same scaler that was used on the training set. We cannot get a new scaler.
   
   point2: we do not apply feature scaling on dummy variables because they are already in the range of [-3,3]
###### Encoding categorical Data

For example consider the  data set that contains one column with categories,for example, France, Spain, or Germany. First, you might guess that it'll be difficult
for machine learning model to compute some correlations between these columns, you know, the features and the outcome, which is the dependent variable.And therefore, of course we'll have to turn these strings I mean the categories into numbers. And this thing that we can do better is actually one hot encoding and one hot encoding consists of turning this country column into three columns. Why three columns? Because there are actually three different classes in this country column, you know, three different categories. If there were, for example, five countries here, we would turn this column into five columns. Let me explain this right away.
So very simply, France would, for example have the vector 1 0 0. Spain would have the vector 0 1 0 and Germany would have the vector 0 0 1. So that then there is not a numerical order between the three countries because instead of having zero, one, and two, we would only have zeros and ones. And therefore, three new columns.

**That is called one hot encoding and that is a very useful and popular method to use when pre-processing your data sets containing categorical variables.**
 
   ###### Encoding the independant variable
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct=ColumnTransformer(transformer[{'encoder',OneHotEncoder(),[0])}, remainder='passthrough')
    x=np.array(ct.fit_transformer(x))
    
   ###### Encoding the dependant variable
      from sklearn.preprocessing import LabelEncoder
      le=LabelEncoder()
      y=le.fit_transform(y)
    
   **Taking care of missing data**
   
   **Removing Missing Value**
   
Generally you don't want to have any missing data in your data set, for the simple reason that it can cause some errors when training your machine learning model, and therefore you must handle them. There are actually several ways to handle them, a first way is to just ignore the observation by deleting it.
That's one method, and this actually works if you have a large data set, and you know if you have only 1% missing data, you know removing 1% of the observations won't change much the learning quality of your model, so 1% is fine, but sometimes you can have a lot of missing data, and therefore you must handle them the right way.

**Replacing by Average**

Now a second way is to actually replace the missing data, the missing value by the average of all the values in the column in which the data is missing. This is a classic way of handling missing data. 

So the next step is indeed to apply this imputer object on the matrix of features. So how are we going to do that?
Well, remember that a class contains an assemble of instructions but also some operations and actions which you can apply to other objects or variables. And these are called methods. You know, they're like functions and one of them is exactly the **fit method**
The fit method will exactly connect this imputer to the matrix of features. In other words, what this fit method will do is it will look at the missing values in selected column and also it will compute the average values of that columns.

All right, so that's simply what it will do. And then that will not be enough to do the replacement we want.
To do the replacement we'll have to call another method called **transform** and which will this time apply the transformation meaning it will replace the missing value by the average of the values.

  ###### Missing values
    from sklearn.impute import simpleimputer
    imputer=simpleimputer(missing_values=np.nan, strategy='mean')
    imputer.fit(selected columns)
    imputer.transform(selected columns)
   

**caution**

Be careful, this transform method actually returns the new updated version of the matrix features with replacements of the missing values. And therefore, what we want to do now and that's the last we have to do is to indeed update our matrix of features 

selected columns=imputer.fit(selected columns)


 
   ###### Import Libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
   
   ###### Import Dataset
    dataset=pd.read_csv('url')

    
