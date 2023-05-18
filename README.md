# Machine-Learning

Here is the list of topics we will be covering in this tutorial:

* [Data preprocessing](#data-preprocessing)

  - [Spliting Data?](#spliting-data)
  - [Feature Scaling?](#feature-scaling)
  - [Encoding categorical Data](#encoding-categorical-data)
  - [Missing Data](#missing-data)

* [Supervised learning](#supervised-learning)

* [Regression](#regression)

* [Unsupervised learning](#supervised-learning)

* [Clustering](#clustering)

   - [Kmean algorithm](#kmean-algorithm)
   
    - [Elbow method](#elbow-method)
 
 * [Deep Learning](#deep-learning)

  - [What is_Deep_Learning?](#what-is-deep-learning)
  - [Deep_Learning_Sections?](#deep-learning-sections)
  - [Encoding categorical Data](#encoding-categorical-data)
  - [Missing Data](#missing-data)
    
        
     
   
    
# Data Preprocessing
## Spliting Data
   In this section we import the data, clean it and we split the data in to training and test set. Usually about 20% of the data is used for test set and 80% for training set. We'll use our training set to build the model. Then we will take the data from the test set and apply our model to them. So they(test data) haven't been part of the model creation process.The model has no information about test data. And the model predicting certain values. But the good news is that because this is something we separated in advance as part of the data that was given to us we actually know the actual values. So now we can compare the predicted values which were generated using a model that has never seen test data before to the actual values. And so from that we can evaluate our model.
   
   ###### Splitting the dataset
      from sklearn.model_selection import train_test_split
      x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
   
 ## Feature Scaling
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
## Encoding categorical Data

For example consider the  data set that contains one column with categories. For example a country column with three values: France, Canada, USA. First, you might guess that it'll be difficult
for machine learning model to compute some correlations between these columns. You know, the features and the outcome, which is the dependent variable.And therefore, of course we'll have to turn these strings, I mean the categories into numbers. And this thing that we can do better is actually one hot encoding that consists of turning this country column into three columns. Why three columns? Because there are actually three different classes in this country column, three different categories. If there were, for example, five countries here, we would turn this column into five columns. Let me explain this right away.
So very simply, France would, for example have the vector 1 0 0. Canada would have the vector 0 1 0 and USA would have the vector 0 0 1. So that then there is not a numerical order between the three countries because instead of having zero, one, and two, we would only have zeros and ones. And therefore, three new columns.

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
    
## Missing Data
   
   **Removing Missing Value**
   
Generally you don't want to have any missing data in your data set, for the simple reason that it can cause some errors when training your machine learning model, and therefore you must handle them. There are actually several ways to handle them, a first way is to just ignore the observation by deleting it.
That's one method, and this actually works if you have a large data set, if you have only 1% missing data,removing them won't change much the learning quality of your model, so 1% is fine, but sometimes you can have a lot of missing data, and therefore you must handle them the right way.

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

# Supervised learning

The way supervised learning algorithms work is that you already have some training data and answers in that training data that you supply to the model.
So, for example, in the case of classification, you have input data, which could be images of apples and you have annotations explaining these are apples
or labels for these apples. And then you supply that to the model. You ask the model to learn from this data with answers. And then you can supply a new image and ask, "What is this?" And it will provide the answer saying that this is an apple.

# Regression

   Regression models (both linear and non-linear) are used for predicting a real value. Let's have a look at simple linear regression equation.
   
   ![Regression Equation](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*GSAcN9G7stUJQbuOhu0HEg.png)
   
   
   
   We will look at the parts of this equation one by one. So on the left, we have our dependent variable, which we're trying to predict.
   On the right, we have our independent variable, which is the predictor. Here we have b0, which is the constant, and b1 is the slope coefficient.
   
   ###### Simple Linear Regression
     from sklearn.linear_model import LinearRegression
     reg=LinearRegression()
     reg.fit(x_train,y_train)
  
  ###### Predicting the test set result
     y_pred=reg.predict(x_test)
     
 
 Question 1: How do I use my simple linear regression model to make a single prediction, for example, to predict the salary of an employee with 12 years of experience?
   ###### Question 1
     print(regressor.predict([[12]]))

Question 2: How do I get the final regression equation y = b0 + b1 x with the final values of the coefficients b0 and b1?

  ###### Question 2
    print(regressor.coef_)
    print(regressor.intercept_)

 
 # Unsupervised learning
 
 Now, unsupervised learning is different. Here, we don't have answers, and the model has to think for itself. So, for example, we might have input data for these images without any labels, supply them to the model and ask them to group these fruits into different categories, even though we don't know the category,
we don't supply the categories. So the machine has no understanding which is an apple, which is a banana, and so on. It can just see that there are certain similarities in the data, certain differences in the data, and from that it can make conclusions and create its own groups. And this is all the while it doesn't understand what it is looking at, doesn't understand the term apple or banana. So, that is the difference between supervised and unsupervised learning. In a nutshell, in supervised learning, you give the model an opportunity to train where it has the answers. In unsupervised learning, you don't have the answers to supply to the model.

 # Clustering
 
 Unsupervised machine learning
 
 Grouping unlabled data
 
Clustering is similar to classification, but the basis is different. In Clustering you don’t know what you are looking for, and you are trying to identify some segments or clusters in your data. When you use clustering algorithms on your dataset, unexpected things can suddenly pop up like structures, clusters and groupings you would have never thought of otherwise.

## Kmean algorithm

Consider we have a scatter plot of our data points and we want K-Means Clustering to create clusters. So we don't have any classes or categories in advance, we don't have any training data. We just have this data and we want to create the clusters.
Well, the first thing is that you need to decide how many clusters you want. Let's say, we decided on two clusters. Then, for each cluster you need to place a randomly centriod on the scatter plot, wherever you like. It doesn't have to be one of the existing points. Now, what happens next is K-Means will assign each of the data points to the closest centroid. Now, the next step is quite interesting. We need to calculate the centre of mass for each of the clusters, the preliminary clusters that we've identified. Of course, the centroid is not included in this calculation. So for example, we need to take all the X coordinates and take the average, and take all the Y coordinates, take the average. That'll give us the position of the center of mass. And then we move the centroid to those positions.
Once they've moved, we repeat the process, we reassign data points to the closest centroid. Reassign, calculate the center of mass, move the centroids, do the process again. And until we get into situation where doing the process again doesn't change anything. So that means we've come to the end of the K-Means Clustering step by step process.

## Elbow method

The question is how do we decide how many clusters to select? Well, the elbow method is one of the approaches to help you make this decision. So the elbow method requires us to look at the equation for the Within Cluster Sum of Squares, or the WCSS. It basically looks at the distance between each point and the centroid of its cluster, square that distance and add them up. 

![43191elbow_img (1)](https://github.com/atenash/Machine-Learning-A-Z/assets/123215187/3720494f-117a-44b0-b65e-762249292578)


To calculate all these different within cluster sum of squares for the different options, we actually need the clusters to already exist. So every time, we have to first run the k-means clustering algorithm, and then we calculate the WCSS. So it's kind of a bit backwards. We don't first do the elbow method to find the optimal number of clusters and then do k-means.

We do k-means many times, find the WCSS for every single setup, whether it's one cluster, 2, 3, 4, 5, and so on,and then we will be able to apply the elbow method. And the second thing to note is that the more clusters we have,the smaller WCSS becomes.
 
 The elbow method  actually a visual method. when you look at this chart and you look for where is the kink in this chart, where is the elbow. And so that is your optimal number of clusters, basically when the WCSS stops dropping as rapidly.
 
  ###### elbow method
     from sklearn.cluster import KMeans
     wcss=[]
     for i in range(,):
        kmeans=KMeans(n_clusters=i, init='k-means++', random_state=)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
     plt.plot(range(,),wcss)
     reg=LinearRegression()
     reg.fit(x_train,y_train)
     
     
   ###### Training the Kmeans model on dataset
     from sklearn.cluster import KMeans
     kmeans=KMeans(n_clusters=n(based on elbow method result), init='k-means++', random_state=)
     y_kmeans=kmeans.fit_predict(x)
     

# Deep Learning
## What is Deep Learning?

Jeffrey Hinton  is known as a godfather of deep learning. The idea behind deep learning is to look at the human brain. We don't know everything about the human brain. But that little amount that we know, we want to mimic it and recreate it. And why is it that? Well, because the human brain seems to be the one of the most powerful tools on this planet for learning, adapting skills, and applying them. And if computers could copy that, then we can just leverage what natural selection has. Like re-inventing the bicycle. A neurons are like a body, they have branches. In the human brain, there's approximately a hundred billion neurons altogether.
But, nevertheless, there's a hundred billion neurons in the human brain and each neuron is connected to as many as thousand of its neighbors. And that's what we're going to try to recreate.

So how do we recreate this in a computer? Well, we recreate an artificial structure called an artificial neural network. Where we have neurons, some of them for input value. For example, you want to predict something, you're always going to have some input. That's called the input layer. Then you have the output, so that's value that you want to predict. Is somebody going to leave the bank or stay in the bank? Is this a fraudulent transaction or is this a real transaction? So that's going to be output layer. And in between, we're going to have a hidden layer. So, as you can see in your brain you have so many neurons. Some information is coming through your eyes, ears, nose. Then it's not just going straightly to the output where you have the result. It's going through all of these billions and billions and billions of neurons before it gets to the output. So we need these hidden layers that are there before the output. The input layer neurons are connected to the hidden layer. Neurons in the hidden layers are connected to the output value.Where is the deep learning here? Why is it called deep learning? Because we have not just one hidden layer, we have lots and lots and lots of hidden layers. And then we connect everything just like in the human brain. Connect everything, inter connect everything and now we're talking deep learning. So that's what deep learning is all about on a very abstract level.

![deep learning.jpeg](https://github.com/atenash/Machine-Learning-A-Z/blob/main/deep%20learning.jpeg)



## Deep Learning Sections

The Neuron

It is the basic building block of artificial neuron networks. Our first challenge is to recreate a neuron. The neuron gets some input signals and it has an output signal. The input signal  are in fact independent variables. The Output value can be continuous. Like price. It can be binary. For instance, a person will exit or will stay or it can be a categorical variable. And if it's a categorical variable the important thing to remember here is that in that case your output value won't be just one it'll be several output values because these will be your dummy variables which will be representing your categories.

![Neuron 1.jpeg](https://github.com/atenash/Machine-Learning-A-Z/blob/main/Neuron%201.jpeg)



Next thing that I want to talk about is weights. Weights are crucial to our artificial neural networks functioning because neural networks learn by adjusting the weights. The neural network decides in every single case what signal is important, what signal is not important to a certain neuron. What signal gets passed along and what signal doesn't, or to what strength, to what extent signals get passed along. So weights are crucial. They are the things that get adjusted through the process of learning. Like when you're training out into artificial neural network, you're basically adjusting all of the weights in all the synapses across this whole neural network. And that's where gradient descent and back propagation come into play.

So signals go into the neuron and what happens in the neuron. Just add up, multiply by the weight and add them up.And then it applies an activation function.

![Neuron 2.jpeg](https://github.com/atenash/Machine-Learning-A-Z/blob/main/Neuron%202.jpeg)


The Activation Function

it's basically a function that is assigned to this neuron or to this whole layer and it is applied to this weighted sum. And then from that, the neuron understands if it needs to pass on a signal. The structure of one neuron has some inputs, values coming in, it's got some weights then it adds up the weighted,it calculates the weighted and then applies the activation function. And on step three, it passes on the signal to the next neuron.

Threshold Function

![threshold function.jpeg](https://github.com/atenash/Machine-Learning-A-Z/blob/main/threshold%20function.jpeg)


On the X-axis, you have the weighted sum of inputs. On the Y-axis, you have the values from zero to one. And basically, the threshold function is a very simple type of function where if the value is less than zero then the threshold function passes on zero. If the value is more than zero or equal to zero then threshold function passes on a one.


Sigmoid Function


![sigmoid.jpeg](https://github.com/atenash/Machine-Learning-A-Z/blob/main/sigmoid.jpeg)


There is 1 divided by 1 plus e to the power of -x. Whereas, x is the value of the sum of the weighted sums. This sigmoid function is very useful in the final layer, in the output layer, especially when you are trying to predict probabilities.


The rectifier function


![Rectifier.jpeg](https://github.com/atenash/Machine-Learning-A-Z/blob/main/Rectifier.jpeg)

Is one of the most popular functions for artificial neural network. So it goes all the way to zero, it is zero. And then from there, it gradually progresses as the input value increases as well.


Hyperbolic Tangent


![hyperbolic.jpeg](https://github.com/atenash/Machine-Learning-A-Z/blob/main/hyperbolic.jpeg)
