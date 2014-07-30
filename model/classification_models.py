import logging
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc

import cPickle as pickle

from utils import plot_ROC, plot_ROC_style, print_classification_report

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

def all_pairs_model(df, classes, CONFIG):
    '''
    Classification for a dataset with 3 classes. Usine all 
    possible pairs, generates threee models.

    Keyword arguments:
        df : DataFrame
          DataFrame with a label columns, specifying the class
          for that row. 
        classes: tuple
          tuple with all class names.
        CONFIG: Namespace
          A namespace with the configurations need to build the
          the model and specifations on the output. 
    Returns:
        None
    '''
    models = {"Logistic":LogisticRegression,
                "RandomForest":RandomForestClassifier,
                "ExtraTrees":ExtraTreesClassifier,
                "GradBoost":GradientBoostingClassifier}

    if CONFIG.show_roc:
        fig, ax = plt.subplots()

    for c1, c2 in combinations(classes, 2):
        data = df[(df.label == c1) | (df.label == c2)]

        X = data.drop("label", 1).values
        classes = {l:i for i,l in enumerate(set(data.label))}
        y = np.array([classes[i] for i in data.label])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        if CONFIG.model_type == "Logistic":
            final_model = models[CONFIG.model_type]()
        else:
            final_model = models[CONFIG.model_type](**CONFIG.model_args)

        final_model.fit(X_train,y_train)
        print "Labels: ", classes
        print "-done training model"

        if CONFIG.serialize_model:
            with open('Parkinson_%s.model'%"_vs_".join(classes.keys()), 'wb') as fw:
                fw.write(pickle.dumps(final_model))
            print "--done serializing model"
        if CONFIG.show_roc:
            y_pred_probas = final_model.predict_proba(X_test)
            plot_ROC(y_pred_probas, y_test, classes, ax)

        print_classification_report(final_model,X_train, X_test, y_train, y_test)

    if CONFIG.show_roc:
        plot_ROC_style(ax)
        plt.show()

def two_class_model(df, classes, CONFIG):
    '''
    classification for a dataset of 3 classes. Usine One vs One,
    and taking pairwise combinations generate three models.

    Keyword arguments:
        df : DataFrame
          DataFrame with a label columns, specifying the class
          for that row. 
        classes: dict 
          contails the class label as key and the numeric label
          as keys.
        CONFIG: Namespace
          A namespace with the configurations need to build the
          the model and specifations on the output. 
    Returns:
        None
    '''
    models = {"Logistic":LogisticRegression,
                "RandomForest":RandomForestClassifier,
                "ExtraTrees":ExtraTreesClassifier,
                "GradBoost":GradientBoostingClassifier}

    X = df.drop("label", 1).values
    y = np.array([classes[i] for i in df.label])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    if CONFIG.model_type == "Logistic":
        final_model = models[CONFIG.model_type]()
    else:
        final_model = models[CONFIG.model_type](**CONFIG.model_args)

    final_model.fit(X_train,y_train)
    print "Labels: ", classes
    print "-done training model"

    if CONFIG.serialize_model:
        with open('Parkinson_two_class.model', 'wb') as fw:
            fw.write(pickle.dumps(final_model))
        print "--done serializing model"

    print_classification_report(final_model, X_train, X_test, y_train, y_test)

    if CONFIG.show_roc:
        fig, ax = plt.subplots()
        y_pred_probas = final_model.predict_proba(X_test)
        #plt.figure()

        plot_ROC(y_pred_probas, y_test, classes, ax)
        plot_ROC_style(ax)
        plt.show()
