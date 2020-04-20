import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

dat=pd.read_csv('D:\Machine Learning\P14-Logistic-Regression\Social_Network_Ads.csv')
dat.replace(to_replace='Male',value=2,inplace=True)
dat.replace(to_replace='Female',value=1,inplace=True)


#Selecting Data
y=dat['Purchased']
dat.drop(['Purchased'],axis=1,inplace=True)

#Splitting Data
X_train,X_test,y_train,y_test=train_test_split(dat,y,stratify=y,test_size=0.30)
lgreg=LogisticRegression(n_jobs=-1,random_state=42,C=1000,penalty='l1',fit_intercept=True)
lgreg.fit(X_train,y_train)
y_pred=lgreg.predict(X_test)

#Confusion Matrix
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)
#Visualizin Cnf Matrix
class_names=[0,1]
fig,ax=plt.subplots()
tick_marks=np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

#Creating Heatmap
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap="YlGnBu",fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix',y=1.1)
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()

#Evaluation Metrics
print('Accuracy:',metrics.accuracy_score(y_test,y_pred))
print('Precision:',metrics.precision_score(y_test,y_pred))
print('Recall:',metrics.recall_score(y_test,y_pred))

#ROC Curve
y_pred_proba=lgreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()