#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Import Packages ##
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, classification_report,accuracy_score,confusion_matrix, roc_curve, auc, roc_curve,accuracy_score,plot_confusion_matrix, roc_auc_score
from sklearn.metrics import recall_score, precision_score, classification_report,accuracy_score,confusion_matrix, roc_curve, auc, roc_curve,accuracy_score,plot_confusion_matrix,make_scorer
from sklearn import svm 
from statistics import mean, stdev
from numpy import interp


# In[2]:


## Read CSV file ##
df = pd.read_csv('file location', delimiter=',',encoding='latin-1')


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(4,8))
colors = ["CASE", "CONTROL"]
sns.countplot('Status', data=df, palette = "Set1")
plt.title('Class Distributions \n (0: Parkinsons disease || 1: Healthy Control)', fontsize=14)


# In[ ]:


## Separate dependent and independent variable 


# In[5]:


y, X = df["Status"], df.drop(columns=["Status","Sample_Id"])


# In[ ]:


## Encode Labels ##


# In[8]:



le = LabelEncoder()
Y = le.fit_transform(y)
Y1=pd.DataFrame(Y)
Y1.columns = ['Output']


# In[14]:


df_new = pd.DataFrame(df.drop(['Status','Sample_Id'], axis=1))


# In[15]:


df_new.head()


# In[20]:


#LASSO


# In[21]:


sel = SelectFromModel(LogisticRegression(C = 1, penalty = 'l1', solver='liblinear'))
#LogisticRegression(C=1, penalty='l1', solver='liblinear')
sel.fit(df_new, Y)


# In[22]:


df_new.shape


# In[23]:


# Number of selected features
selected_feat = df_new.columns[(sel.get_support())]


# In[24]:


type(selected_feat)


# In[25]:


print('total features: {}'.format((df_new.shape[1])))


# In[26]:


print('selected features: {}'.format(len(selected_feat)))


# In[27]:


print(selected_feat)


# In[28]:


# number of parameters rejected
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel.estimator_.coef_ == 0)))


# In[29]:


## create dataset with selected features
x_l1 = sel.transform(df_new)


# In[30]:


x_l1=pd.DataFrame(x_l1, columns=selected_feat)


# In[31]:


x_l1.head()


# In[32]:


x_l1.shape


# In[33]:


##### Build ML model and compare performance
# https://github.com/krishnadulal/Feature-Selection-in-Machine-Learning-using-Python-All-Code/blob/master/Embedded%20Feature/Recursive%20Feature%20Selection%20by%20Using%20Tree%20Based%20and%20Gradient%20Based%20Estimators%20.ipynb


# In[35]:



from sklearn.metrics import recall_score, precision_score, classification_report,accuracy_score,confusion_matrix, roc_curve, auc, roc_curve,accuracy_score,plot_confusion_matrix,make_scorer


# In[36]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb


# In[37]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import model_selection


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x_l1,Y,test_size=0.15, stratify=Y)


# In[39]:


x_train.head()


# In[40]:


x_train.shape


# In[41]:


### Combine x_l1 and Y #####


# In[42]:


strtied_dataset1 = [x_l1, Y1]

final_df1 = pd.concat(strtied_dataset1, sort=False, axis=1)


# In[43]:


final_df1.head()


# In[44]:


print(final_df1.shape)


# In[45]:


### StratifiedKFold----- SVM  ###


# In[49]:


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
skf = StratifiedKFold(n_splits=10)
lst_accu_stratified=[]
lst_precision_stratified=[]
lst_recall_stratified=[]
lst_roc_auc_stratified=[]


model = svm.SVC(C=0.25, kernel='linear', gamma='auto',probability=True) #C=1, kernel='linear',probability=True
i = 0
def train_model(train, test, fold_no):
   # X = ['Retail_Price','Discount']
    #y = ['Returned_Units']
    x_train = train.drop(['Output'],axis=1)
    y_train = train.Output
    x_test = test.drop(['Output'],axis=1)
    y_test = test.Output
    
    model.fit(x_train,y_train)
    predictions = model.predict(x_test)
    print('Fold',str(fold_no),'Accuracy:',accuracy_score(y_test,predictions),)
    lst_accu_stratified.append(accuracy_score(y_test,predictions))
    lst_precision_stratified.append(precision_score(y_test,predictions))
    lst_recall_stratified.append(recall_score(y_test,predictions))
    lst_roc_auc_stratified.append(roc_auc_score(y_test,predictions))
    
    fpr, tpr, thresholds = roc_curve(y_test,predictions)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (fold_no, roc_auc))
    #i += 1
    
fold_no = 1
for train_index, test_index in skf.split(final_df1, Y):
   # train = dataset.iloc[train_index,:]
   # test = dataset.iloc[test_index,:]
    train = final_df1.loc[train_index,:]
    test = final_df1.loc[test_index,:]
    train_model(train,test,fold_no)
    fold_no += 1
    
print('\nOverall Accuracy:', mean(lst_accu_stratified)*100, '%')
print('\nStandard Deviation Accuracy is:', stdev(lst_accu_stratified))
print('\nOverall precision:', mean(lst_precision_stratified)*100, '%')
print('\nStandard Deviation Precision is:', stdev(lst_precision_stratified))
print('\nOverall Recall:', mean(lst_recall_stratified)*100, '%')
print('\nStandard Deviation Recall is:', stdev(lst_recall_stratified))
print('\nOverall roc_auc:', mean(lst_roc_auc_stratified)*100, '%')
print('\nStandard Deviation roc_auc is:', stdev(lst_roc_auc_stratified))


# In[53]:


plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',  alpha=.8)#label='Chance',

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'SVM-Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.title('ROC-AUC (SVM-LASSO)',fontsize=14)
plt.legend(loc="lower right", prop={'size': 10})
plt.show()

