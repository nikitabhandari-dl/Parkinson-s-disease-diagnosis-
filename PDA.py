## Import Packages ##
import pandas as pd
import numpy as np
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
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import metrics
from sklearn.metrics import recall_score, precision_score, classification_report,accuracy_score,confusion_matrix, roc_curve, auc, roc_curve,accuracy_score,plot_confusion_matrix, roc_auc_score, make_scorer
from sklearn import svm 
from statistics import mean, stdev
from numpy import interp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn import model_selection




## Read CSV file ##
df = pd.read_csv('file location', delimiter=',',encoding='latin-1')
plt.figure(figsize=(4,8))
colors = ["CASE", "CONTROL"]
sns.countplot('Status', data=df, palette = "Set1")
plt.title('Class Distributions \n (0: Parkinsons disease || 1: Healthy Control)', fontsize=14)


## Separate dependent and independent variable 
y, X = df["Status"], df.drop(columns=["Status","Sample_Id"])
## Encode Labels ##
le = LabelEncoder()
Y = le.fit_transform(y)
Y1=pd.DataFrame(Y)
Y1.columns = ['Output']
df_new = pd.DataFrame(df.drop(['Status','Sample_Id'], axis=1))
df_new.head()

#LASSO
sel = SelectFromModel(LogisticRegression(C = 1, penalty = 'l1', solver='liblinear'))
#LogisticRegression(C=1, penalty='l1', solver='liblinear')
sel.fit(df_new, Y)
#Shape of df_new
df_new.shape
# Number of selected features
selected_feat = df_new.columns[(sel.get_support())]
type(selected_feat)
print('total features: {}'.format((df_new.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print(selected_feat)

# number of parameters rejected
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel.estimator_.coef_ == 0)))

# create dataset with selected features
x_l1 = sel.transform(df_new)
x_l1=pd.DataFrame(x_l1, columns=selected_feat)
x_l1.head()
#Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_l1,Y,test_size=0.15, stratify=Y)
x_train.head()
x_train.shape

### Combine x_l1 and Y #####
strtied_dataset1 = [x_l1, Y1]
final_df1 = pd.concat(strtied_dataset1, sort=False, axis=1)
final_df1.head()
print(final_df1.shape)

### StratifiedKFold ###
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
skf = StratifiedKFold(n_splits=10)
lst_accu_stratified=[]
lst_precision_stratified=[]
lst_recall_stratified=[]
lst_roc_auc_stratified=[]


model = ML_technique(parameter's) 
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
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',  alpha=.8)#label='Chance',

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Algorithm-Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.title('ROC-AUC (ML_technique-LASSO)',fontsize=14)
plt.legend(loc="lower right", prop={'size': 10})
plt.show()


#SHAP
import shap
#helps you organize your javascript 
shap.initjs()
# use Kernel SHAP to explain test set predictions
shap_explainer = shap.KernelExplainer(model.predict_proba, x_train_l1, link="logit")   
shap_values = shap_explainer.shap_values(x_test_l1, nsamples=150, l1_reg='num_features(10)')
#Print the length of shap values and each element
print(f'length of shap values -> n_classes: {len(shap_values)}')
print(f'length of each element values: {shap_values[0].shape}')
x_test_l1_df=pd.DataFrame(x_test_l1)
idx = pd.Index(selected_feat)
idx1=idx.tolist()
##Explaining a Single Prediction
print(f'Prediction for 1st sample in test data: {model.predict_proba(x_test_l1_df.iloc[[0],:])[0]}')
shap.initjs()
shap.force_plot(shap_explainer.expected_value[0], shap_values[0][0,:],x_test_l1_df.iloc[0,:], link='logit')  
#Explaining predictions for all samples
shap.initjs()
shap.force_plot(shap_explainer.expected_value[0], shap_values[0], x_test_l1_df)     
shap.initjs()
shap.force_plot(shap_explainer.expected_value[1], shap_values[1], x_test_l1_df)
#Shap summary plot
shap.summary_plot(shap_values, x_test_l1_df)                     
shap.summary_plot(shap_values[0], x_test_l1_df)
                     
