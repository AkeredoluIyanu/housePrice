#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor


# ## Getting Data Ready

# In[2]:


import pandas as pd

dataset = pd.read_csv("Housing_dataset_train.csv")
test = pd.read_csv("Housing_dataset_test.csv")
sub = pd.read_csv("Sample_submission.csv")


# ## A look at the top five rows using the DataFrame’s head() method

# In[3]:


dataset.head()


# In[4]:


test.head()


# In[5]:


sub.head()


# In[ ]:





# ## A quick description of the data, in particular the total number of rows, and each attribute’s type and number of non-null values¶

# In[6]:


dataset.info()


# There are 14,000 instances in the train dataset. Notice that the loc, title, bedroom, bathroon and parking space doesn't have up to 14,000 entries, meaning that some districts are missing this feature.
# 
# All attributes are numerical, except the loc and title attributes. Its type is object and I know that it must be a text attribute. I find out what categories exist and how many districts belong to each category by using the value_counts() method:

# In[7]:


dataset["loc"].value_counts()


# In[8]:


dataset["title"].value_counts()


# In[ ]:





# ## A summary of the numerical attributes

# In[9]:


dataset.describe()


# In[10]:


import matplotlib.pyplot as plt
dataset.hist(bins=50, figsize=(20,15))
plt.show()


# A histogram for each numerical attribute

# In[ ]:





# ## Handling Null Values in  Columns

# In[11]:


# Deleting the row with missing(NAN) data
dataset.dropna(inplace=True)


# In[12]:


# Checing if there is still missing data
dataset.isnull().sum()


# In[ ]:





# ## Handling Text and Categorical Attributes
# Earlier I left out the categorical attribute loc and area because it is a text
# attribute so I cannot compute their summary of A numerical attribute. Most Machine Learning algorithms pre‐
# fer to work with numbers anyway, so let’s convert these text labels to numbers.
# Scikit-Learn provides a transformer for this task called LabelEncoder:

# In[13]:


# Python code to convert categorical attribute to numerical attribute using LabelEncoder

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)       


# In[ ]:





# In[14]:


dataset=MultiColumnLabelEncoder(columns = ['loc','title']).fit_transform(dataset)
dataset


# In[ ]:





# In[15]:


test=MultiColumnLabelEncoder(columns = ['loc','title']).fit_transform(test)
test


# In[ ]:





# A look at how much each attribute correlates with the median house value
# The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that
# there is a strong positive correlation and when the coefficient is close to –1, it means
# that there is a strong negative correlation

# In[16]:


corr_matrix = dataset.corr()
corr_matrix["price"].sort_values(ascending=False)


# In[ ]:





# ## Discover and Visualize the Data to Gain Insights
# A quick glance at the data to get a general understanding of
# the kind of data I am manipulating. Now the goal is to go a little bit more in depth.
# 

# In[17]:


import matplotlib.pyplot as plt
dataset.hist(bins=50, figsize=(20,15))
plt.show()


# This image tells you that the housing prices are very much related to the bedroom attribute than the remaing attributes

# In[ ]:





# In[18]:


# A look at the dataset again
dataset


# In[ ]:





# ## Select and Train a Model
# I framed the problem, I have got the data, clean it and explored it. I
# prepare the data for Machine Learning algorithms. I am now ready
# to select and train a Machine Learning model

# In[19]:


X=dataset.drop(["price"], axis=1)
y=dataset["price"]


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
 X, y, random_state=0)


# In[ ]:





# ## CatBoostRegressor Model

# In[24]:


from catboost import CatBoostRegressor
cat_reg = CatBoostRegressor()
cat_reg.fit(X_train, y_train)

car_predictions = cat_reg.predict(X_train)
cat_mse = mean_squared_error(y_train, car_predictions)
cat_rmse = np.sqrt(cat_mse)
cat_rmse


# In[ ]:





# ## checinking the test set accuracy and the mean squared error

# In[26]:


print("Test set accuracy: {:.2f}".format(cat_reg.score(X_test, y_test)))
cat_pred = cat_reg.predict(X_test)

print(f'mse = {mean_squared_error(y_test, cat_pred, squared=False)}')


# In[27]:


# A quick glance at the submision file
sub


# In[30]:


predictions = cat_reg.predict(test)
predictions


# ## Adding the predictions in a new column to the Submition file

# In[31]:


sub['price'] = predictions
sub.head()


# In[32]:


sub.head()


# ## Saving the DataFrame as a csv file

# In[ ]:


sub.to_csv('cbrHouse_price model.csv', index=False)


# In[ ]:




