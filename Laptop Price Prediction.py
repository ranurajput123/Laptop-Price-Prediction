#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


# In[7]:


df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv')


# In[8]:


df


# In[9]:


df.head()


# In[10]:


df.shape


# In[11]:


df.info()


# In[12]:


# no two rows are same
df.duplicated().sum()


# In[13]:


# no missing value
df.isnull().sum()


# Preprocessing
# -

# - Removing Unnamed column,GB from RAM,Kg from wieght and converting float to integer.

# In[16]:


df.drop(columns=['Unnamed: 0'],inplace=True)


# In[17]:


df.head()


# In[35]:


df['Ram'].str.replace('GB','')


# In[37]:


df['Weight'].str.replace('kg','')


# In[38]:


df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')


# In[39]:


df.head()


# In[40]:


df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')


# In[41]:


df.info()


# EDA
# -

# In[49]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[51]:


sns.distplot(df['Price'])
plt.show()


# - Above data is skewed.

# Company column
# -

# In[53]:


# number of selling by each brand
df['Company'].value_counts().plot(kind='bar')
plt.show()


# In[54]:


# average cost of each brand
sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# Type Name column
# -

# In[56]:


df['TypeName'].value_counts().plot(kind='bar')
plt.show()


# In[57]:


sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# Inches column
# -

# In[61]:


sns.distplot(df['Inches'])
plt.show()


# - Max. no. of laptops which are sold belongs to 15.6-16 inches

# In[63]:


sns.scatterplot(x=df['Inches'],y=df['Price'])
plt.show()


# Screen Resolution column
# -

# In[64]:


df['ScreenResolution'].value_counts()


# In[65]:


df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)


# In[68]:


df.sample(5)


# In[69]:


df['Touchscreen'].value_counts().plot(kind='bar')
plt.show()


# - Touchscreen laptops are very low in quantity.

# In[30]:


sns.barplot(x=df['Touchscreen'],y=df['Price'])


# In[31]:


df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[32]:


df.head()


# In[33]:


df['Ips'].value_counts().plot(kind='bar')


# In[34]:


sns.barplot(x=df['Ips'],y=df['Price'])


# In[35]:


new = df['ScreenResolution'].str.split('x',n=1,expand=True)


# In[36]:


df['X_res'] = new[0]
df['Y_res'] = new[1]


# In[37]:


df.sample(5)


# In[38]:


df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[39]:


df.head()


# In[40]:


df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')


# In[41]:


df.info()


# In[42]:


df.corr()['Price']


# In[43]:


df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')


# In[44]:


df.corr()['Price']


# In[45]:


df.drop(columns=['ScreenResolution'],inplace=True)


# In[46]:


df.head()


# In[47]:


df.drop(columns=['Inches','X_res','Y_res'],inplace=True)


# In[48]:


df.head()


# CPU column
# -

# In[49]:


df['Cpu'].value_counts()


# In[50]:


df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[51]:


df.head()


# In[52]:


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[53]:


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)


# In[54]:


df.head()


# In[55]:


df['Cpu brand'].value_counts().plot(kind='bar')


# In[56]:


sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[57]:


df.drop(columns=['Cpu','Cpu Name'],inplace=True)


# In[58]:


df.head()


# Ram column
# -

# In[59]:


df['Ram'].value_counts().plot(kind='bar')


# In[60]:


sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# Memory column
# -

# In[61]:


df['Memory'].value_counts()


# In[62]:


df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


# In[63]:


df.sample(5)


# In[64]:


df.drop(columns=['Memory'],inplace=True)


# In[65]:


df.head()


# In[66]:


df.corr()['Price']


# In[67]:


df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)


# In[68]:


df.head()


# GPU column
# -

# In[69]:


df['Gpu'].value_counts()


# In[70]:


df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])


# In[71]:


df.head()


# In[72]:


df['Gpu brand'].value_counts()


# In[73]:


df = df[df['Gpu brand'] != 'ARM']


# In[74]:


df['Gpu brand'].value_counts()


# In[75]:


sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()


# In[76]:


df.drop(columns=['Gpu'],inplace=True)


# In[77]:


df.head()


# Operating System column
# -

# In[78]:


df['OpSys'].value_counts()


# In[79]:


sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[80]:


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[81]:


df['os'] = df['OpSys'].apply(cat_os)


# In[82]:


df.head()


# In[83]:


df.drop(columns=['OpSys'],inplace=True)


# In[84]:


sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# Weight column
# -

# In[85]:


sns.distplot(df['Weight'])


# In[86]:


sns.scatterplot(x=df['Weight'],y=df['Price'])


# In[87]:


df.corr()['Price']


# In[88]:


sns.heatmap(df.corr())


# In[89]:


sns.distplot(np.log(df['Price']))


# In[90]:


X = df.drop(columns=['Price'])
y = np.log(df['Price'])


# In[91]:


X


# In[93]:


y


# In[94]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)


# In[95]:


X_train


# In[96]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error


# In[118]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
#from xgboost import XGBRegressor


# Linear Regression
# -

# In[119]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# Ridge Regression
# -

# In[99]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# Lasso Regression
# -

# In[100]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# KNN
# -

# In[101]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# Decision Tree
# -

# In[102]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# SVM
# -

# In[103]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# Random Forest
# -

# In[108]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# Ada Boost
# -

# In[109]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=15,learning_rate=1.0)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# Gradient Boost
# -

# In[114]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[120]:


df


# In[ ]:




