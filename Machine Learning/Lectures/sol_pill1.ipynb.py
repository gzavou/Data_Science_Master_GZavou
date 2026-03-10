#!/usr/bin/env python
# coding: utf-8

# # The machine learning pipeline
# 
# Modeling churn means to understand what keeps the customer engaged to our product. Its analysis goal is to predict or describe the **churn rate** i.e. the rate at which customer leave or cease the subscription to a service. Its value lies in the fact that engaging new customers is often more costly than retaining existing ones. For that reason subscription business-based companies usually have proactive policies towards customer retention.
# 
# In this case study, we aim at building a machine learning based model for customer churn prediction on data from a Telecom company. Each row on the dataset represents a subscribing telephone customer. Each column contains customer attributes such as phone number, call minutes used during different times of day, charges incurred for services, lifetime account duration, and whether or not the customer is still a customer.
# 
# This case is partially inspired in Eric Chiang's analysis of churn rate. Data is available from the University of California Irvine machine learning repositories data set.

# ## Goal
#  + Implement a full machine learning pipeline.
#  + Understand the concepts of training, validation, and test.

# In[2]:


import pandas as pd

dl=pd.read_csv(r'C:\Users\ilari\OneDrive\Desktop\MASTER UB\MACHINE LEARNING\Pill1\First Steps (4)\files\churn_curated_numerical.csv',header=None)


# In[3]:


dl.head()


# In[4]:


data = dl.values


# In[5]:


data.shape


# In[6]:


X = data[:,:-1]
y = data[:,-1]


# In[7]:


X.shape


# In[8]:


import numpy as np
np.unique(y)


# In[9]:


X


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
plt.pie(np.c_[len(y)-np.sum(y),np.sum(y)][0],labels=['No Churn','Churn'],colors=['r','g'],shadow=True,autopct ='%.2f' )
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.plot()


# ## Data
# 
# Observe data

# In[11]:


import numpy as np

np.mean(X,axis=0)
print(np.var(X,axis=0))


# A problem in Scikit-Learn is modeled as follows:
# 
# + Input data is structured in Numpy arrays. The size of the array is expected to be [n_samples, n_features]:
# 
#     + *n_samples*: The number of samples ($N$): each sample is an item to process (e.g. classify). A sample can be a document, a picture, a sound, a video, an astronomical object, a row in database or CSV file, or whatever you can describe with a fixed set of quantitative traits.
#   
#     + *n_features*: The number of features ($d$) or distinct traits that can be used to describe each item in a quantitative manner. Features are generally real-valued, but may be boolean, discrete-valued or even cathegorical.
# 
# $${\rm feature~matrix:} {\bf X}~=~\left[
# \begin{matrix}
# x_{11} & x_{12} & \cdots & x_{1d}\\
# x_{21} & x_{22} & \cdots & x_{2d}\\
# x_{31} & x_{32} & \cdots & x_{3d}\\
# \vdots & \vdots & \ddots & \vdots\\
# \vdots & \vdots & \ddots & \vdots\\
# x_{N1} & x_{N2} & \cdots & x_{Nd}\\
# \end{matrix}
# \right]$$
# 
# $${\rm label~vector:} {\bf y}~=~ [y_1, y_2, y_3, \cdots y_N]$$
#     
# 
# The number of features must be fixed in advance. However it can be very high dimensional (e.g. millions of features) with most of them being zeros for a given sample. 

# Create and fit a decision tree (you can find it in the module sklearn.tree and the name is DecisionTreeClassifier)

# In[12]:


from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X,y)


# Predict the data you used for training/fiting the classifer

# In[13]:


yhat = clf.predict(X)


# In[14]:


yhat[100],y[100]


# <div class="alert alert-success" style = "border-radius:10px"><b>EXERCISE:</b> We need a measure of how well the classifier is performing. `yhat` is a list and our target outcome `y`is also a list. Create a measure of accuracy. </div>

# In[ ]:


#Your code here


# One sensible way of measuring the goodness of a classifier is measuring the error rate, i.e. the number of times the classifier fails divided by the total number of elements.
# 
# $$err = \frac{1}{N}\sum \mathbb{1}_{\tilde{y}!=y}$$
# 
# where $\mathbb{1}_{\text{cond}}$ is the indicator function given a condition, $\text{cond}$, defined as 
# 
# $$\mathbb{1}_{\text{cond}}=\left \{\begin{align} 1 & \quad\text{if cond = True}\\ 0 & \quad\text{otherwise} \end{align}\right.$$
# Alternatively, we can report the accuracy, defined as the rate of success
# 
# $$acc = \frac{1}{N}\sum \mathbb{1}_{\tilde{y}==y}.$$
# 
# Observe that $acc = 1-err$.
# 
# `sklearn` reports this result using the method from module `metrics`, `.accuracy_score`.

# In[15]:


from sklearn import metrics

metrics.accuracy_score(y,yhat)


# <div class = "alert alert-info" style="border-radius:10px"> <b>QUESTION:</b> Is this a good result?</div>

# <div class = "alert alert-warning" style="border-radius:10px">  BACK TO SLIDES!!!</div>

# # Knowledge representation

# In[16]:


#Load data set.
from sklearn import datasets
digits = datasets.load_digits()


# In[17]:


#Check the data format.
X, y = digits.data, digits.target

print (X.shape)
print (y.shape)


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
# The original digit has been flattened, so we reshape it back to its original form
# Check the dimensionality of the data, e.g. the first element in the data set X[0]
print (X[0].shape)
print (X[0])

# Reshape it to 8x8 to recover the original image
print (X[0].reshape((8,8)))


# Show the image using scikit.image package
import matplotlib.pyplot as plt

plt.imshow(X[2].reshape((8,8)),cmap="gray",interpolation="nearest")
plt.show()


# In[19]:


#Visualize some of the data.
import matplotlib.pyplot as plt
fig, ax = plt.subplots(8, 12, subplot_kw={'xticks':[], 'yticks':[]})
for i in range(ax.size):
    ax.flat[i].imshow(digits.data[i].reshape(8, 8),
                      cmap=plt.cm.binary)
fig.set_size_inches((10,6))
plt.show()


# <div class = "alert alert-info" style="border-radius:10px"> <b>DISCUSS THE FOLLOWING KNOWLEDGE REPRESENTATION:</b> 
# <p>    
# (A) We are asked to develop a product for automatic translation of text from a document. 
# <p>
# (B) We are asked to develop a product similar to Shazzam(tm). This is, recognize the name of a song given a small sample of the music.*
# <p>
# Discuss and describe a posible feature vector for this problem with your partner.
# </div>

# ### More intuition about the data: The feature space
# 
# Data is usually gathered as raw values. In the case of the digits dataset the gray values of the image. However, we can use domain knowledge we may consider important in order to discriminate the different classes. Take for instance two very simple derived features: horizontal, vertical symmetry and area. 

# In[20]:


from skimage import io as io

tmp = X[7].reshape((8,8))    
sym = tmp*tmp[:,::-1]
io.imshow(tmp)
io.show()
io.imshow(tmp[:,::-1])
io.show()

import numpy as np
Xnew = np.zeros((y.shape[0],3))
for i in range(y.shape[0]):
    area = sum(X[i])
    tmp = X[i].reshape((8,8))    
    symH = tmp*tmp[:,::-1]
    symV = tmp*tmp[::-1,:]
    
    Xnew[i,:]=[sum(symH.flatten()), area, sum(symV.flatten())]

print (Xnew)
print (Xnew.shape)


# In[ ]:


import matplotlib.pyplot as plt
idxA = y==0
idxB = y==6

feature1 = 0
feature2 = 1


plt.figure()
plt.scatter(Xnew[idxA, feature1], Xnew[idxA,feature2], c='green',alpha=0.8)
plt.scatter(Xnew[idxB, feature1], Xnew[idxB,feature2], c='red',alpha=0.8)
plt.show()


# <div class = "alert alert-success"> **Exercise** Change feature1 and feature2 axis $\in \{0,1,2\}$ and select the most suitable view for classification purposes. Why did you select that view?
# </div>

# The process of using knowledge domain information in order to create discriminant features is called <span style="color:red">feature extraction</span>.

# #### Raw data vs feature extraction
# 
# **Raw data**
# 
# Advantages:
# 
# + No domain specific knowledge is required.
# 
# Drawbacks:
# 
# + Highly redundant in many cases and usually span very large dimensional spaces.
# + Unknown discriminability.
# 
# **Feature extraction**
# 
# Advantages:
# 
# + Attempt to capture discriminant information in the data.
# + Lower dimensionality and complexity.
# 
# Drawbacks: 
# 
# + Domain specific knowledge is required.

# <div class = "alert alert-info" style="border-radius:10px"> <b>ACTION:</b> BACK TO SLIDES</div>

# # Generalization

# In[21]:


import pandas as pd

dl=pd.read_csv(r'C:\Users\ilari\OneDrive\Desktop\MASTER UB\MACHINE LEARNING\Pill1\First Steps (4)\files\churn_curated_numerical.csv',header=None)
data = np.asarray(dl)
X = data[:,:-1]
y = data[:,-1]


# In[22]:


from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X,y)
yhat = clf.predict(X)


# In[23]:


from sklearn import metrics

metrics.accuracy_score(y,yhat)


# Let us change the model, and check what we obtain with a different classifier.

# In[24]:


from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X,y)
yhat = clf.predict(X)
metrics.accuracy_score(y,yhat)


# <div class = "alert alert-info" style="border-radius:10px"> <b>QUESTION:</b> This is a pretty good result, isn't it?</div>

# <img src = "./files/i-dont-know-rick-it-looks-fake.jpg">

# <div class = "alert alert-info" style="border-radius:10px"> <b>QUESTION:</b> Is this the value we expect to have when we apply this method in production?</div>

# In real applications we will train a classifier on a given data set but then apply the classifier to unseen data. Let us simulate this process by spliting the data set in two sets. We will call data we use for fiting the classifier training and data used for assessing the performance, test data.

# <div class="alert alert-success" style = "border-radius:10px"><b>EXERCISE:</b> Split the data set 70% for training purposes and the rest for test purposes. You should end up with four variables `X_train`, `y_train`, `X_test`, `y_test`. <p>
# <b>OTHER REQUIREMENTS:</b> Reshuffle data using a permutation of the indexes (`np.random.permutation(...)`) and set the seed of the random number generator using `np.random.seed(42)`
# </div>

# In[ ]:


#Your code here


# Split data in training and set, use the module cross_validation, train_test_split , use random_state=42 as an argument for reproductibility.

# In[25]:


from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, random_state=42)



# In[27]:


from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
yhat = clf.predict(X_train)

from sklearn import metrics

metrics.accuracy_score(y_train,yhat)

yhat_test = clf.predict(X_test)
print(metrics.accuracy_score(y_test,yhat_test))


# Let us try a new algorithm, nearest neighbor, with parameter n_neighbors = 1

# In[28]:


from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=1)

clf.fit(X_train,y_train)
yhat = clf.predict(X_train)
from sklearn import metrics

metrics.accuracy_score(y_train,yhat)


# In[29]:


yhat = clf.predict(X_test)
from sklearn import metrics

metrics.accuracy_score(y_test,yhat)


# <div class = "alert alert-info" style="border-radius:10px"> <b>QUESTION:</b> Is this a good result?</div>

# In[ ]:


#INSERT SNOOPING CODE


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, random_state=42)

#X_train_scaled = scaler.fit_transform(X_train)

scaler = scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)


clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train_scaled,y_train)
yhat = clf.predict(X_train_scaled)
from sklearn import metrics
 
metrics.accuracy_score(y_train,yhat)

X_test_scaled = scaler.transform(X_test)

yhat = clf.predict(X_test_scaled)
from sklearn import metrics

metrics.accuracy_score(y_test,yhat)


# In[30]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train_scaled,y_train)
yhat = clf.predict(X_train_scaled)
from sklearn import metrics

print(metrics.accuracy_score(y_train,yhat))


# In[31]:


from sklearn.preprocessing import StandardScaler

X_test_scaled = scaler.transform(X_test)

yhat = clf.predict(X_test_scaled)
from sklearn import metrics

print(metrics.accuracy_score(y_test,yhat))


# This result is ok, but how consistent is it? Maybe we have been lucky with the train-test partion. We can repeat this process for different values of the random_state (or just use random permutations) and report the average result.

# <div class="alert alert-success" style = "border-radius:10px"><b>EXERCISE:</b> We want a vector of accuracies, `acc`, of shape (10,1) with the values of testing a 1-NN classifier on a `train_size=0.7` for the different random_states stored in the array `r_state`. Use `sklearn.cross_validation.train_test_split` function.
# </div>

# In[ ]:


r_state = [0,1,2,3,4,5,42,43,44,45]


# In[ ]:


#Your code here


# In[34]:


acc


# Check your code with the following visualization code:

# In[33]:


import matplotlib.pyplot as plt
fig = plt.figure()
plt.boxplot(acc)
plt.scatter(np.ones((1,acc.shape[0]))+0.01*np.random.normal(size=(1,10)),acc,alpha = 0.5,color='r')
plt.show()


# <div class = "alert alert-info" style = "border-radius:10px"> <b>QUIZ:</b> Report the average accuracy.</div>

# In[ ]:


#Your code


# In[32]:


#My code
value = np.mean(acc)


# In[ ]:


assert np.abs(value-0.8691)<0.0001


# <div class = "alert alert-info" style="border-radius:10px"> <b>ACTION:</b> BACK TO SLIDES</div>

# # Model selection (I)

# We have tried a 1-Nearest Neighbors classifiers but we could try also different values for the Nearest Neighbors. The selection of a model between different alternatives is called model selection. We can use the same strategy as before and report accuracies for the three models. Let us do it comparintg 1-NN, 3-NN and a DecisionTree.

# <div class="alert alert-success" style = "border-radius:10px"><b>EXERCISE:</b> We want a matrix of accuracies, `acc`, of shape (10,3) with the values of testing a decision tree, a 1-NN, and a 3-NN classifier on a `train_size=0.7` for the different random_states stored in the array `r_state`. 
# </div>

# In[35]:


#My code

from sklearn import metrics
from sklearn import tree
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler

r_state = [0,1,2,3,4,5,42,43,44,45]

acc = np.zeros((len(r_state),3))

for i in range(len(r_state)):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, random_state=r_state[i])
    
    #Your code here
    
    acc[i,0] = metrics.accuracy_score(y_test,yhat_tr)
    acc[i,1] = metrics.accuracy_score(y_test,yhat_nn1)
    acc[i,2] = metrics.accuracy_score(y_test,yhat_nn3)


# In[38]:


#My code

from sklearn import metrics
from sklearn import tree
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler

r_state = [0,1,2,3,4,5,42,43,44,45]

acc = np.zeros((len(r_state),3))

for i in range(len(r_state)):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, random_state=r_state[i])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    

    tr = tree.DecisionTreeClassifier()
    tr.fit(X_train_scaled,y_train)    
    nn1 = neighbors.KNeighborsClassifier(n_neighbors=1)
    nn1.fit(X_train_scaled,y_train)
    nn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    nn3.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.transform(X_test)
    
    yhat_tr = tr.predict(X_test_scaled)
    yhat_nn1 = nn1.predict(X_test_scaled)
    yhat_nn3 = nn3.predict(X_test_scaled)
    
    acc[i,0] = metrics.accuracy_score(y_test,yhat_tr)
    acc[i,1] = metrics.accuracy_score(y_test,yhat_nn1)
    acc[i,2] = metrics.accuracy_score(y_test,yhat_nn3)
print(acc)


# In[37]:


import matplotlib.pyplot as plt
fig = plt.figure()
plt.boxplot(acc)
plt.scatter(np.tile(np.array([1,2,3]),(10,1))+0.01*np.random.normal(size=(10,3)),acc,alpha = 0.5,color='r')
plt.show()


# <div class = "alert alert-info" style = "border-radius:10px" ><b>QUIZ:</b > What is the best of the three methods?</div>

# <div class = "alert alert-info" style = "border-radius:10px"><b>QUIZ:</b> What is the expected accuracy of the selected method in exploitation over unseen data? </div>

# <div class="alert alert-danger" style = "border-radius:10px"><b>EXERCISE:</b> The  `breast_cancer` dataset from `datasets` (check `load_breast_cancer`) reports a set of clinical trials with the outcome of breast cancer detection. We want to build a method to predict whether a patient has a potential cancer or not according to her clinical trials.
# 
# <p>
# 
# For that purpose we will use two different models, a support vector machine and a gradient boosting machine. We will train different settings of a support vector machine (`svm.SVC`) and a single gradient boosting machine (`ensemble.GradientBoostingMachine`) with the following parameters:
# 
# <ul>
# <li>
# `svm.SVC(C=10.0,gamma = 1e-5,random_state=42)`
# </li>
# <li>
# `svm.SVC(C=100.0,gamma = 1e-5,random_state=42)`
# </li>
# <li>
# `svm.SVC(C=1000.0,gamma = 1e-6,random_state=42)`
# </li>
# <li>
# `ensemble.GradientBoostingClassifier(random_state=42)`
# </li>
# </ul>
# <p>
# For selection purposes and accuracy evaluation we will use `model_selection.train_test_split`.
# The data set will be divided using parameters `test_size = 100` and `random_state=42`. As a result of this first division we will have a big training set and a 100 samples test set. Following that and using the same settings we will divide the remaining training set into the final training set and the validation set with 100 samples again.
# 
#     
# <p>
#     DO NOT PREPROCESS OR NORMALIZE DATA!
# <p>
# 
# Prepare the following answers:
# 
# <ol>
# <li>
# Check the sizes of the training, validation and test sets.
# </li>
# <li>
# Report the training accuracy of all four methods.
# </li>
# <li>
# Report the validation accuracy for all methods.
# </li>
# <li>
# Report the performance of all methods using the test set.
# </li>
# </ol>
# 
# <p>
# In the light of the answers obtained from the exercise:
# <ul>
# <li>
# Question 1: What is the size of the training set? 
# </li>
# <li>
# Question 2: What method do you select?
# </li>
# <li>
# Question 3: What is the expected performance of the method selected?
# </li>
# </ul>
# 
# </div>

# In[86]:


#Your code
#Load data set.
from sklearn import datasets
data=datasets.load_breast_cancer()

#Check the data format.
X, y = data.data, data.target

print (X.shape)
print (y.shape)


# In[87]:


from sklearn import metrics
from sklearn import svm
from sklearn import ensemble

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 100, random_state=42)

X_train_final, X_val, y_train_final, y_val = model_selection.train_test_split(X_train, y_train, test_size=100, random_state=42)

svm1 = svm.SVC(C=10.0,gamma = 1e-5,random_state=42)
svm2 = svm.SVC(C=100.0,gamma = 1e-5,random_state=42)
svm3 = svm.SVC(C=1000.0,gamma = 1e-6,random_state=42)
gbm = ensemble.GradientBoostingClassifier()


# In[88]:


print("Training set size:", X_train_final.shape)  # Dimensione del training set finale
print("Validation set size:", X_val.shape)        # Dimensione del validation set
print("Test set size:", X_test.shape)


# In[89]:


import numpy as np
acc = np.zeros((4,1))

svm1 = svm.SVC(C=10.0,gamma = 1e-5,random_state=42)
svm1.fit(X_train_final,y_train_final)
yhat_tr1 = svm1.predict(X_train_final)
yhat_val1 = svm1.predict(X_val)
yhat_test1 = svm1.predict(X_test)
a_tr_1 = metrics.accuracy_score(y_train_final,yhat_tr1)
a_val_1 = metrics.accuracy_score(y_val,yhat_val1)
a_test_1 = metrics.accuracy_score(y_test,yhat_test1)


svm2 = svm.SVC(C=100.0,gamma = 1e-5,random_state=42)
svm2.fit(X_train_final,y_train_final)
yhat_tr2 = svm2.predict(X_train_final)
yhat_val2 = svm2.predict(X_val)
yhat_test2 = svm2.predict(X_test)
a_tr_2 = metrics.accuracy_score(y_train_final,yhat_tr2)
a_val_2 = metrics.accuracy_score(y_val,yhat_val2)
a_test_2 = metrics.accuracy_score(y_test,yhat_test2)


svm3 = svm.SVC(C=1000.0,gamma = 1e-6,random_state=42)
svm3.fit(X_train_final,y_train_final)
yhat_tr3 = svm3.predict(X_train_final)
yhat_val3 = svm3.predict(X_val)
yhat_test3 = svm3.predict(X_test)

a_tr_3 = metrics.accuracy_score(y_train_final,yhat_tr3)
a_val_3 = metrics.accuracy_score(y_val,yhat_val3)
a_test_3 = metrics.accuracy_score(y_test,yhat_test3)


gbm = ensemble.GradientBoostingClassifier(random_state=42)
gbm.fit(X_train_final,y_train_final)
yhat_trgbm = gbm.predict(X_train_final)
yhat_valgbm = gbm.predict(X_val)
yhat_testgbm = gbm.predict(X_test)

a_tr_gbm = metrics.accuracy_score(y_train_final,yhat_trgbm)
a_val_gbm = metrics.accuracy_score(y_val,yhat_valgbm)
a_test_gbm = metrics.accuracy_score(y_test,yhat_testgbm)



acc = np.zeros((4,3))
acc[0,0] = a_tr_1
acc[1,0] = a_tr_2
acc[2,0] = a_tr_3
acc[3,0] = a_tr_gbm
acc[0,1] = a_val_1
acc[1,1] = a_val_2
acc[2,1] = a_val_3
acc[3,1] = a_val_gbm
acc[0,2] = a_test_1
acc[1,2] = a_test_2
acc[2,2] = a_test_3
acc[3,2] = a_test_gbm

print(acc)

# yhat_svm2 = nn1.predict(X_train)
# yhat_svm3 = nn3.predict(X_train)
# yhat_gbm = tr.predict(X_train)


# In[ ]:





# In[ ]:





# In[ ]:




