
# coding: utf-8

# In[17]:

# Import packages
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sea
get_ipython().magic('matplotlib inline')


# In[18]:

# Import Data

training = pd.DataFrame(pd.read_csv('data_train.csv', delimiter=' *, *', engine='python'))
test = pd.DataFrame(pd.read_csv('data_test.csv', delimiter=' *, *', engine='python'))
del test['Unnamed: 15']
del test['Unnamed: 16']

# Remove full stop from Income in Test
test['income'] = test['income'].map(lambda x: str(x)[:-1])

# Combine datasets

data = pd.concat([training,test])


# In[19]:

# Convert all meausres to binary for K-Means clustering

# Remove edu-num as the same data as education

del data['edu-num']
del data['fnlwgt']

# Group categories with low counts

# Combine Categories in Education
data['education'].replace(['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th'],['Primary', 'Primary', 'Primary','Lower-HS', 'Lower-HS', 'Upper-HS','Upper-HS','Upper-HS',], inplace=True)
data['education'].replace(['Masters','Doctorate'],['Post-Grad','Post-Grad'], inplace=True)

# Combine Occupation

data['occupation'].replace(['Prof-specialty','Exec-managerial','Adm-clerical','Sales','Protective-serv','Priv-house-serv','Craft-repair','Machine-op-inspct','Transport-moving','Handlers-cleaners','Farming-fishing','Armed-Forces','?'],['Prof-Exec', 'Prof-Exec', 'Admin-Sales','Admin-Sales', 'Other-service', 'Other-service','Manual-work','Manual-work','Manual-work','Manual-work','Manual-work','Other-service','Unknown'], inplace=True)

# Combine Workclass
data['workclass'].replace(['Self-emp-not-inc','Self-emp-inc','Local-gov','State-gov','Federal-gov','Without-pay','Never-worked','?'],['Self-Emp','Self-Emp','Gov','Gov','Gov','Other','Other','Other'], inplace=True)
#Combine Countries

data['country'].replace(['Mexico','Puerto-Rico','El-Salvador','Cuba','Jamaica','Dominican-Republic','Guatemala','Columbia','Haiti','Nicaragua','Peru',     'Honduras','Trinadad&Tobago','Ecuador','Outlying-US(Guam-USVI-etc)'],['Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib',], inplace=True)
data['country'].replace(['Philippines','India','China','Vietnam','Japan','Taiwan','Hong','Cambodia','Thailand','Laos',],['Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia',], inplace=True)
data['country'].replace(['Germany','England','Italy','Poland','Portugal','France','Greece','Ireland','Yugoslavia','Hungary','Scotland','Holand-Netherlands'],['Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe',], inplace=True)
data['country'].replace(['?','South','Iran'],['Other','Other','Other'], inplace=True)

#Convert numerical/continuous data to binary for clustering
data.loc[data['capital-gain'] > 0, 'capital-gain'] = 1
data.loc[data['capital-loss'] > 0, 'capital-loss'] = 1

#Convert ages to groups
count = 8
for i in range(65,0,-10):
    data.loc[data.age > i, 'age'] = count
    count -=1

#Convert hours to >=40, under 40

data.loc[data['hours'] > 40, 'hours'] = 3
data.loc[data['hours'] > 30, 'hours'] = 2
data.loc[data['hours'] > 5, 'hours'] = 1


# In[39]:

print(data['age'])


# In[20]:

# Convert categorical vars to binary dummies
bindata = pd.get_dummies(data,columns=['hours','age','income','sex', 'workclass','education','occupation','marital','relationship','race','country'])


# In[21]:

# Plot correlations
plt.subplots(figsize=(20,20))
sea.heatmap(bindata.corr(), square=True)
plt.show()


# In[22]:

headers = list(bindata.columns.values)
bind = np.asarray(bindata)


# In[23]:

#Perform clustering - determine optimal number of clusters by silhouette score

# Not working due to machine's memory limitations

#for i in range (1,11):
 #   km = KMeans(n_clusters=i, n_init=50)
 #   km.fit(bind)
  #  labels = km.labels_
#    print("Score for " + str(i) + " Clusters: " + str(metrics.silhouette_score(bind, labels, metric='euclidean')))


# In[24]:

# Perform clustering with single cluster value

km = KMeans(n_clusters=6, n_init=50, init='k-means++')
km.fit(bind)


# In[26]:

#Extract labels, append to original data set
labels = km.labels_
headers.append('cluster')
clusters = np.column_stack((bind,labels))
clusters = pd.DataFrame(clusters)
clusters.columns = headers


# In[ ]:




# In[27]:

#Find numner of data points per feature in each cluster
cls = clusters.groupby('cluster').sum()
clscount = clusters.groupby('cluster').count()
cls['total'] = clscount['capital-gain']
print(cls)


# In[28]:

# Find % of total by rows per feature and cluster
cls_perc = pd.DataFrame()
for col in cls:
    cls_perc[col] = cls[col]/cls['total']
print(cls_perc)


# In[29]:

for col in cls_perc:
    print (str(col) + " Highest %: " + str(cls_perc[col].argmax(1)))


# In[30]:

# Find % of total by column per feature and cluster
cls_p2 = pd.DataFrame()
for col in cls:
    cls_p2[col] = cls[col]/cls[col].sum()
print(cls_p2)


# In[31]:

# Plot by % of rows
plt.subplots(figsize=(20,5))
sea.heatmap(cls_perc)
plt.show()


# In[32]:

# Plot by % of cols
plt.subplots(figsize=(20,5))
sea.heatmap(cls_p2)
plt.show()

