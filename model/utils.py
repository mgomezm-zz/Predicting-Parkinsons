import logging

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc

import cPickle as pickle

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
        filename="history.log", filemode='a', level=logging.DEBUG,
        datefmt='%m/%d/%y %H:%M:%S')

formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
        datefmt='%m/%d/%y %H:%M:%S')

console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logger = logging.getLogger(__name__)

def plot_ROC(probas, y_true, classes, ax):
    '''
    Given prediction probabilities, and true labes, plot an ROC curve

    Kyeword arguments:
        *ax* : tuple
          single axis object
    Returns:
        None
    '''

    fpr, tpr, thresholds = roc_curve(y_true, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    ax.plot(fpr, tpr,linewidth=2.0, label='AUC = %0.2f, %s' %(roc_auc, " vs ".join(classes.keys())))
    #ax.fill_between(fpr, 0, tpr, alpha=.2)

def plot_ROC_style(ax):
    '''
    Styles the ROC figure

    Kyeword arguments:
        *ax* : tuple
          single axis object
    Returns:
        None
    '''
    ax.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC curve', fontsize=18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.legend(loc="lower right", prop={'size':18})

def print_classification_report(model, X_train, X_test, y_train, y_test):
    '''
    '''
    y_pred = model.predict(X_test)
    print model
    print "Train score: ", model.score(X_train, y_train)
    print "Test score: ", model.score(X_test, y_test)
    print "Classification Report: "
    print classification_report(y_pred, y_test)
    print 

