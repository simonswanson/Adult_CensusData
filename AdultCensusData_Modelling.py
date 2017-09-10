
# coding: utf-8

# In[1]:

# Import packages
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn import metrics
get_ipython().magic('matplotlib inline')


# In[2]:

# Import Data

training = pd.DataFrame(pd.read_csv('data_train.csv', delimiter=' *, *', engine='python'))
test = pd.DataFrame(pd.read_csv('data_test.csv', delimiter=' *, *', engine='python'))
del test['Unnamed: 15']
del test['Unnamed: 16']
train.head()
test.head()


# In[3]:

for name in training:
    print((training[name].dtype))


# In[4]:

# Investigate Data

# Check target variable balance in each dataset

print("Training % Over $50K: " + str(training[training.income == '>50K'].count()['income']/len(training))) 
print("Test % Over $50K: " + str(test[test.income == '>50K.'].count()['income']/len(test))) 

# Print counts of values per variable
for name in training.columns:
    print ("Name: " + str(name) + "\n")
    print (training[name].value_counts())
    


# In[5]:

# Print charts per variable   

fig = plt.figure(figsize = (20,30))
for count, name in enumerate(training.columns,1):
    fig = plt.subplot(5,3,count)
    if training[name].dtype == np.int64:
        plt.title(name)
        plt.hist(training[name])
    else:
        plt.title(name)
        training[name].value_counts().plot(kind="bar")
plt.tight_layout()
plt.show()        


# In[6]:

#Clean up data

# Remove edu-num as the same data as education
del training['edu-num']
del test['edu-num']

# Remove relationship due to correlation with marital and sex
del training['relationship']
del test['relationship']

# Remove fnlwgt due to lack of categorizability
del training['fnlwgt']
del test['fnlwgt']

# Remove rows with unknown values
training = training[(training.workclass != '?') & (training.occupation != '?') & (training.country != '?')]
test = test[(test.workclass != '?') & (test.occupation != '?') & (test.country != '?')]



# In[7]:

# Group categories with low counts by similar features

# Combine Categories in Education
training['education'].replace(['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th'],['Primary', 'Primary', 'Primary','Lower-HS', 'Lower-HS', 'Upper-HS','Upper-HS','Upper-HS',], inplace=True)
training['education'].replace(['Masters','Doctorate'],['Post-Grad','Post-Grad'], inplace=True)
test['education'].replace(['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th'],['Primary', 'Primary', 'Primary','Lower-HS', 'Lower-HS', 'Upper-HS','Upper-HS','Upper-HS',], inplace=True)
test['education'].replace(['Masters','Doctorate'],['Post-Grad','Post-Grad'], inplace=True)
#print (training['education'].value_counts())

# Combine Occupation

training['occupation'].replace(['Prof-specialty','Exec-managerial','Adm-clerical','Sales','Protective-serv','Priv-house-serv','Craft-repair','Machine-op-inspct','Transport-moving','Handlers-cleaners','Farming-fishing','Armed-Forces','?'],['Prof-Exec', 'Prof-Exec', 'Admin-Sales','Admin-Sales', 'Other-service', 'Other-service','Manual-work','Manual-work','Manual-work','Manual-work','Manual-work','Other-service','Unknown'], inplace=True)
test['occupation'].replace(['Prof-specialty','Exec-managerial','Adm-clerical','Sales','Protective-serv','Priv-house-serv','Craft-repair','Machine-op-inspct','Transport-moving','Handlers-cleaners','Farming-fishing','Armed-Forces','?'],['Prof-Exec', 'Prof-Exec', 'Admin-Sales','Admin-Sales', 'Other-service', 'Other-service','Manual-work','Manual-work','Manual-work','Manual-work','Manual-work','Other-service','Unknown'], inplace=True)
#print (training['occupation'].value_counts())

# Combine Workclass
training['workclass'].replace(['Self-emp-not-inc','Self-emp-inc','Local-gov','State-gov','Federal-gov','Without-pay','Never-worked','?'],['Self-Emp','Self-Emp','Gov','Gov','Gov','Other','Other','Other'], inplace=True)
test['workclass'].replace(['Self-emp-not-inc','Self-emp-inc','Local-gov','State-gov','Federal-gov','Without-pay','Never-worked','?'],['Self-Emp','Self-Emp','Gov','Gov','Gov','Other','Other','Other'], inplace=True)
#print (training['workclass'].value_counts())

#Combine Countries

training['country'].replace(['Mexico','Puerto-Rico','El-Salvador','Cuba','Jamaica','Dominican-Republic','Guatemala','Columbia','Haiti','Nicaragua','Peru',     'Honduras','Trinadad&Tobago','Ecuador','Outlying-US(Guam-USVI-etc)'],['Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib',], inplace=True)
training['country'].replace(['Philippines','India','China','Vietnam','Japan','Taiwan','Hong','Cambodia','Thailand','Laos',],['Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia',], inplace=True)
training['country'].replace(['Germany','England','Italy','Poland','Portugal','France','Greece','Ireland','Yugoslavia','Hungary','Scotland','Holand-Netherlands'],['Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe',], inplace=True)
training['country'].replace(['?','South','Iran'],['Other','Other','Other'], inplace=True)
test['country'].replace(['Mexico','Puerto-Rico','El-Salvador','Cuba','Jamaica','Dominican-Republic','Guatemala','Columbia','Haiti','Nicaragua','Peru',     'Honduras','Trinadad&Tobago','Ecuador','Outlying-US(Guam-USVI-etc)'],['Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib','Lat-Am-Carib',], inplace=True)
test['country'].replace(['Philippines','India','China','Vietnam','Japan','Taiwan','Hong','Cambodia','Thailand','Laos',],['Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia',], inplace=True)
test['country'].replace(['Germany','England','Italy','Poland','Portugal','France','Greece','Ireland','Yugoslavia','Hungary','Scotland','Holand-Netherlands'],['Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe',], inplace=True)
test['country'].replace(['?','South','Iran'],['Other','Other','Other'], inplace=True)


# In[8]:

#Replace ages with age groups
count = 8
for i in range(65,0,-10):
    training.loc[training.age > i, 'age'] = count
    count -=1

count = 8
for i in range(65,0,-10):
    test.loc[test.age > i, 'age'] = count
    count -=1


# In[9]:

# Convert target tp 1 and 0, create separate array as y, convert to integers, remove from main dataset

y_train = np.array(training['income'].replace(['<=50K','>50K'],['0','1']).astype(int))
y_test = np.array(test['income'].replace(['<=50K.','>50K.'],['0','1']).astype(int))
del training['income']
del test['income']


# In[10]:

# Convert categorical vars to binary vector dummies
xtrain = pd.get_dummies(training,columns=['age','workclass','education','occupation','marital','sex','race','country'])
xtest = pd.get_dummies(test,columns=['age','workclass','education','occupation','marital','sex','race','country'])


# In[11]:

# Show correlation plot
plt.subplots(figsize=(20,20))
sea.heatmap(xtrain.corr())
plt.show()


# In[12]:

# Save column headers for later use with array, check if any values present in training set but not in test set
headers = xtrain.columns.values
missing = xtest.columns.values
print(set(headers) - set(missing))


# In[13]:

#tranform to arrays
x_train = np.asarray(xtrain)
x_test = np.asarray(xtest)
print(x_train.shape)
print(x_test.shape)


# In[23]:

# Scale Data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# In[18]:

# Define function to test classifier models with K-Fold X-Validation

def clf_tester(X,y,model, name, metrics, folds):
    clf = []
    clf.append((name,model))
    pipeline = Pipeline(clf)
    kfold = KFold(n_splits=folds)
    results = cross_val_score(pipeline, X, y, cv=kfold, scoring=metrics)      
    return results



# In[19]:

logreg = LogisticRegression(penalty='l1')
lda = LinearDiscriminantAnalysis(solver='svd')
svm = SVC(kernel="rbf")
nnet = MLPClassifier(hidden_layer_sizes=(200,),solver='sgd')
rfc = RandomForestClassifier(n_estimators=50)
knn = KNeighborsClassifier(n_neighbors = 4)
nb = GaussianNB()


# In[100]:

# Test classifiers with K-Fold X-validation

# Test algorithms with default settings

mets = ['accuracy','recall']
logreg = LogisticRegression(penalty='l1')
lda = LinearDiscriminantAnalysis(solver='svd')
models = [(logreg,'Logistic Regression'),(lda, 'LDA')]
for model, name in models:
    print('%s :' %name,)
    for metric in mets:
        model_results = clf_tester(x_train,y_train,model, name, metric, 5)
        mean = model_results.mean()
        std = model_results.std()
        print(metric + "\t Mean:" + str(mean) + "\t Std Dev: " + str(std))


# In[29]:

# Check classifier models: LogReg, LDA, SVM, NNEts, RandomForest, KNN, NBayes

mets = ['accuracy','recall']
logreg = LogisticRegression(penalty='l1')
lda = LinearDiscriminantAnalysis(solver='svd')
svm = SVC(kernel="rbf")
nnet = MLPClassifier(hidden_layer_sizes=(200,),solver='sgd')
rfc = RandomForestClassifier(n_estimators=50)
knn = KNeighborsClassifier(n_neighbors = 4)
nb = GaussianNB()
models = [(logreg,'Log_Reg'),(lda, 'LDA'),(svm, 'SVM'),(nnet, 'N_Net'),(rfc, 'Random_Forest'),(knn, 'K-NN'),(nb, 'N_Bayes')]
for model, name in models:
    print('%s :' %name,)
    for metric in mets:
        model_results = clf_tester(x_train,y_train,model, name, metric, 5)
        mean = model_results.mean()
        std = model_results.std()
        print(metric + "\t Mean:" + str(mean) + "\t Std Dev: " + str(std))


# In[27]:




# In[20]:

#Log Reg - check on test data
logreg.fit(x_train,y_train)
ypred = logreg.predict(x_test)

#Neural Network - check on test data
nnet.fit(x_train,y_train)
ypred_nn = nnet.predict(x_test)


# In[28]:

# Define function to get test data set scores

def print_metrics(model, y_test, ypred):
    print (str(model) + "Test Results:\n")
    print ("Accuracy:\t" + str(round(metrics.accuracy_score(y_test, ypred),3)))
    print ("Precision:\t" + str(round(metrics.precision_score(y_test, ypred),3)))
    print ("F1:\t\t" + str(round(metrics.f1_score(y_test, ypred),3)))
    print ("Recall:\t\t" + str(round(metrics.recall_score(y_test, ypred),3)))
    print ("\nConfusion Matrix:\n" + str(metrics.confusion_matrix(y_test,ypred)))

#Print Results for Log Reg, NNet

print_metrics("Logistic Regression", y_test, ypred)
print_metrics("\n\nNeural Network", y_test, ypred_nn)


# In[22]:

print(np.bincount(y_test))


# In[143]:

#Try with different parameters
logreg2 = LogisticRegression(penalty='l2')
logreg2.fit(x_train,y_train)
ypred2 = logreg2.predict(x_test)
print_metrics("Logistic Regression L2", y_test, ypred2)


# In[144]:

print(np.bincount(y_test))

# Test Random Forest with different number of estimators

for i in range(50,500,50):
    rfc = RandomForestClassifier(n_estimators=i)
    rfc.fit(x_train,y_train)
    yptrain = rfc.predict(x_test)
    print_metrics(i, y_test, yptrain)


# In[145]:

print(logreg.intercept_)
coeffs = pd.Series(logreg.coef_[0], index=headers)
coeffs = coeffs.sort_values()
print(coeffs)
plt.figure(figsize=(20,20))
ax = plt.subplot(2,1,2)
coeffs.plot(kind="bar")
plt.show()


# In[149]:

#print(logreg.intercept_)
ldacoeffs = pd.Series(lda.coef_[0], index=headers)
ldacoeffs = ldacoeffs.sort_values()
print(ldacoeffs)
plt.figure(figsize=(20,20))
ax = plt.subplot(2,1,2)
ldacoeffs.plot(kind="bar")
plt.show()


# In[29]:

# Try PCA to detect primary variables

#PCA

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

# Normalize X
xnorm = scale(x_train)


# In[30]:

Xpca = PCA()
xp = Xpca.fit_transform(xnorm)


# In[31]:

Xvar= Xpca.explained_variance_ratio_
Xcomp = Xpca.explained_variance_
print ((Xcomp))
print(Xvar)


# In[32]:


X_cumvar=np.cumsum(np.round(Xpca.explained_variance_ratio_, decimals=4)*100)
plt.plot(X_cumvar)
plt.xlabel("Principal components")
plt.ylabel("Variance captured")
plt.title("Explained Variance by Number of Principal Components")
plt.show()

