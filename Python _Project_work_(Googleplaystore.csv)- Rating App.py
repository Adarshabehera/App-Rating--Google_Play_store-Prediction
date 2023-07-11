#!/usr/bin/env python
# coding: utf-8

# In[128]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[129]:


inp0 =pd.read_csv('googleplaystore.csv')


# In[130]:


inp0 .head()


# In[131]:


inp0 .tail()


# In[132]:


inp0 .isnull().sum(axis=0)


# # Data Preprocessing

# In[133]:


inp0.info()


# In[134]:


inp0.dropna(how = 'any', inplace=True)


# In[135]:


inp0 .isnull().sum(axis=0)


# In[136]:


inp0.dtypes


# In[137]:


inp0.head()


# # Manipulating price data types

# In[138]:


inp0.Price.value_counts()[:5]


# In[139]:


inp0['Price'] = inp0.Price.map(lambda x:0 if x == '0' else float(x[1:]))


# # converting reviews col to integers types

# In[140]:


inp0.Reviews = inp0.Reviews.astype("int32")


# In[141]:


inp0.Reviews.describe()


# # then Here comes the  install columns

# In[142]:


inp0.Installs.value_counts()


# In[143]:


channging the + symbols inbetween the data sets


# In[144]:


def clean_installs(val):
    return int(val.replace(",","").replace("+",""))


# # inp0.Installs = inp0.Installs.map(clean_installs)

# In[145]:


inp0.Installs.describe()


# # Handling the app size

# In[146]:


def change_size(size):
    if'M' in size:
        x = size[:-1]
        x = float(x)*1000
        return(x)
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x)
        return(x)
    else:
        return None


# In[147]:


inp0["Size"] = inp0["Size"].map(change_size)
inp0.Size.describe()                     


# In[148]:


# Filling the NA Values in data of Size


# In[149]:


inp0.Size.fillna(method = 'ffill', inplace = True)


# In[150]:


inp0.dtypes


# # Some Sanity check ups

# In[151]:


inp0.Rating.describe()


# In[152]:


inp0.Rating.head()


# In[153]:


inp0.Rating.tail()


# In[154]:


# For Reviews Activity


# In[155]:


len(inp0[inp0.Reviews > inp0.Installs])


# In[ ]:


inp0[inp0.Reviews > inp0.Installs]


# In[ ]:


inp0 = inp0[inp0.Reviews <= inp0.Installs].copy()


# In[ ]:





# In[156]:


inp0.shape


# In[157]:


len(inp0[(inp0.type =="Free") & (inp0.Price > 0)])


# # Some Basic EDA plot For Visualizations

# In[158]:


sns.boxplot(inp0.Price)
plt.show()


# In[159]:


# For Reviews col


# In[160]:


sns.boxplot(inp0.Reviews)
plt.show()


# # For the Distribution of Ratings

# In[161]:


inp0.Rating.plot.hist()
plt.show()


# In[162]:


#Histogrm of size


# In[163]:


inp0['Size'].plot.hist()
plt.show()


# # Let;s ,manipulate price col

# In[164]:


len(inp0[inp0.Price > 200])


# In[165]:


inp0[inp0.Price > 200]


# In[166]:


inp0 = inp0[inp0.Price <= 200].copy()
inp0.shape


# # Dropping very high installs values

# In[167]:


inp0.Installs.quantile([0.1, 0.25, 0.5, 0.70, 0.90, 0.95, 0.99])


# In[ ]:


len(inp0[inp0.Installs >= 100000000])


# In[168]:


inp0 = inp0[inp0.Installs < 100000000].copy()
inp0.shape


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[169]:


# For Bivarate Analysis (For making Scatter plot /Jointplot Analysis):-


# In[265]:


sns.pairlot(inp0.Price, inp0.Rating)


# In[171]:


sns.jointplot(inp0.Size, inp0.Rating)


# In[172]:


sns.jointplot(inp0.Reviews, inp0.Rating)


# In[173]:


# For Box plot Analysis


# In[176]:


plt.figure(figsize = [8,6])
sns.boxplot(inp0['Content Rating'], inp0.Rating)


# In[177]:


plt.figure(figure = [18,6])
g = sns.boxplot(inp0.Category, inp0.Rating)
plt.xtricks(rotation=90)


# In[179]:


sns.pairplot(inp0)


# In[184]:


sns.boxplot(inp0)


# # PreProcessing step

# In[185]:


inp1 = inp0.copy()


# In[186]:


inp0.Installs.describe()


# In[193]:


inp1.Installs = inp1.Installss.apply(np.log1p)


# In[192]:


inp1.Reviews = inp1.Reviews.apply(np.log1p)


# In[194]:


inp1.dtypes


# In[196]:


inp1.drop(["App","Last Updated","Current Ver","Android Ver"], axis=1, inplace= True)


# In[197]:


inp1.shape


# In[198]:


inp2 = pd.get_dummies(inp1, drop_first = True)


# In[213]:


inp2.columns


# # Test_Train Split 

# In[201]:


from sklearn.model_selection import train_test_split


# In[204]:


df_train,df_test  = train_test_split (inp2, train_size = 0.7, random_state=100)


# In[205]:


df_train.shape


# In[206]:


df_test.shape


# # Determine features and Labels

# In[214]:


inp2.columns


# In[216]:


features=['Reviews', 'Size', 'Price', 'Category_AUTO_AND_VEHICLES',
       'Category_BEAUTY', 'Category_BOOKS_AND_REFERENCE', 'Category_BUSINESS',
       'Category_COMICS', 'Category_COMMUNICATION',
       'Genres_Tools', 'Genres_Tools;Education', 'Genres_Travel & Local',
       'Genres_Travel & Local;Action & Adventure', 'Genres_Trivia',
       'Genres_Video Players & Editors',
       'Genres_Video Players & Editors;Creativity',
       'Genres_Video Players & Editors;Music & Video', 'Genres_Weather',
       'Genres_Word']


# In[218]:


x


# In[220]:


y=inp2.Rating


# In[221]:


x


# In[222]:


y


# # Splitting the dataset(Training & Test data set)

# In[234]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[235]:


x_train


# In[236]:


x_test


# In[238]:


y_train


# In[239]:


y_test


# # Train the algorithm

# In[240]:


from sklearn.linear_model import LinearRegression


# In[241]:


Abde_villi = LinearRegression()


# In[242]:


Abde_villi.fit(x_train, y_train)


# # Predict the test data set (x-test) 

# In[1]:


y_pred = Abde_villi.predict(x_test)


# In[ ]:


y_pred


# In[245]:


y_test


# # Evaluating the model

# In[250]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[251]:


print('MAE', mean_absolute_error(y_test,y_pred))


# In[252]:


print('MSE', mean_squared_error(y_test,y_pred))


# In[253]:


from math import sqrt as sqrt
print('RMSE',sqrt(mean_absolute_error(y_test,y_pred)))


# In[261]:


r2=r2_score(y_test,y_pred)*1000


# In[262]:


print(r2)


# In[ ]:


As we got only 17% as value as our r2 values , it is not fittable for our model  for better analysis,MSE,MAE all 
value should be low for better model building.

