import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize,scale
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn import svm


from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc


import cPickle as pickle
def plot_ROC(probas, y_true, classes):
    '''
    Given prediction probabilities, and true classes, plot an ROC curve
    '''
    fpr, tpr, thresholds = roc_curve(y_true, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)
    #Plot ROC curve
    #plt.plot(fpr, tpr,linewidth=2.0, label='AUC = %0.2f, %s' %(roc_auc, " vs ".join(classes.keys())))

def cross_validate(df):
    '''
    INPUT: pandas DataFrame, numpy ndarray, sklearn estimator
    OUTPUT: N/A
    Creates train-test split, fits model and then shows performance report.
    '''
    plt.figure()
    models = []
    for data in [df[(df.label == "PD") | (df.label == "SWEDD")],
                    df[(df.label == "PD") | (df.label == "HC")],
                    df[(df.label == "HC") | (df.label == "SWEDD")]]:
        X = scale(data.drop("label", 1).values)
        #X = data.drop("label", 1).values
        classes = {l:i for i,l in enumerate(set(data.label))}
        y = np.array([classes[i] for i in data.label])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        final_model = RandomForestClassifier(n_estimators=100)#  compute_importances=True)
        #final_model = ExtraTreesClassifier(n_estimators=200)#,  compute_importances=True)
        #final_model = LogisticRegression()
        final_model.fit(X_train,y_train)
        print "done training model", classes
        y_pred_probas = final_model.predict_proba(X_test)
        plot_ROC(y_pred_probas, y_test, classes)
        #print 100*final_model.feature_importances_
        print
    '''
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC curve', fontsize=18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.legend(loc="lower right", prop={'size':18})

    plt.show()
    '''

    #make predictions
    # print model
    # print "Train score:", model.score(X_train, y_train)
    # print "Test score:", model.score(X_test, y_test)
    # print "Classification Report:", classification_report(y_pred, y_test)
    # print

def two_class_model(df, serialize = False):
    '''
        Simple classifier, where SWEDD is treated as PD.
    '''
    #X = scale(df.drop("label", 1).values)
    X = df.drop("label", 1).values

    classes = {"PD":1, "SWEDD":1, "HC":0}
    y = np.array([classes[i] for i in df.label])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    #final_model = RandomForestClassifier(n_estimators=100)#,  compute_importances=True)
    #final_model = ExtraTreesClassifier(n_estimators=100)#,  compute_importances=True)
    final_model = LogisticRegression()
    final_model.fit(X_train,y_train)
    print "done training model", classes
    y_pred_probas = final_model.predict_proba(X_test)
    plot_ROC(y_pred_probas, y_test, classes)
    #print 100*final_model.feature_importances_
    #print final_model.coef_
    #print
    if serialize:
        with open('Parkinson_two_class.model', 'wb') as fw:
            fw.write(pickle.dumps(final_model))
        print "done serializing model"

if __name__ == "__main__":
    df_status = pd.read_csv("./../data/Patient_Status.csv")[['PATNO', 'ENROLL_CAT']]
    patient_class = dict(df_status.dropna().values)

    df_UPDRSII = pd.read_csv("./../data/MDS_UPDRS_Part_II__Patient_Questionnaire.csv")
    df_UPDRSII = df_UPDRSII[df_UPDRSII.EVENT_ID == "BL"]
    df_UPDRSII = df_UPDRSII[["PATNO","NP2SPCH","NP2SALV","NP2SWAL","NP2EAT","NP2DRES","NP2HYGN","NP2HWRT","NP2HOBB","NP2TURN","NP2TRMR","NP2RISE","NP2WALK","NP2FREZ"]]

    df = df_UPDRSII
    df['label'] = df.PATNO.apply(lambda x: patient_class.get(x, np.nan))
    df = df.dropna() #subset=['label'])
    #df = df.drop("PATNO",1)
    #cross_validate(df)
    two_class_model(df[["label","NP2SPCH","NP2HWRT","NP2TRMR"]], True)
    #two_class_model(df[["label","NP2SPCH","NP2HWRT","NP2TRMR"]])

