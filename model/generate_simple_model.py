import argparse
import logging
import ast
from itertools import combinations

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

from classification_models import all_pairs_model, two_class_model


#for debugging purpose 
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

def main(CONFIG):
    '''
    For this model will only focus on three predictors from
    the Motor assessments datasets:
        (speech, hand writing, tremor).

    Depending on the settings will build a model and provide
    performance metrics. 

    Keyword arguments:
        CONFIG: Namespace
          A namespace with the configurations need to build the
          the model and specifations on the output. 
    Returns:
        None
    '''
    try:
        df_status = pd.read_csv(CONFIG.patient_file)[['PATNO', 'ENROLL_CAT']]
    except IOError:
        print "Error: can\'t find file or read patient file"

    patient_class = dict(df_status.dropna().values)

    try:
        df_UPDRSII = pd.read_csv(CONFIG.data_file)
    except IOError:
        print "Error: can\'t find file or read data file"

    df_UPDRSII = df_UPDRSII[df_UPDRSII.EVENT_ID == "BL"]
    df = df_UPDRSII[["PATNO", "NP2SPCH","NP2HWRT","NP2TRMR"]]

    df['label'] = df.PATNO.apply(lambda x: patient_class.get(x, np.nan))
    df = df.dropna()
    df = df.drop("PATNO",1)

    if CONFIG.multiclass:
        classes = ("PD", "SWEDD", "HC")
        all_pairs_model(df, classes, CONFIG)
    else:
        classes = {"PD":1, "SWEDD":1, "HC":0}
        two_class_model(df, classes, CONFIG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train different models, show ROC curves, and serialize the chosen model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-patient_file',
                        default="./../data/Patient_Status.csv",
                        type=str,
                        required=False,
                        help='CSV with patient info (labels).')

    parser.add_argument('-data_file',
                        default="./../data/MDS_UPDRS_Part_II__Patient_Questionnaire.csv",
                        type=str,
                        required=False,
                        help='CSV with training data.')

    parser.add_argument('-multiclass',
                        default=False,
                        type=bool,
                        required=False,
                        help='Three classes available (HC, SWEDD, PD). Generate model for all'+
                            ' class (all-pairs)? or merge SWEDD and PD together for binary'+
                            ' classifcatoin.')

    parser.add_argument('-model_type',
                        default= 'Logistic',
                        type=str,
                        help='Type of model to fit.',
                        required=False,
                        choices=["Logistic", 
                                "RandomForest",
                                "ExtraTrees",
                                "GradBoost"])

    parser.add_argument('-serialize_model',
                        default=False,
                        type=bool,
                        help='Save model.')

    parser.add_argument('-model_args',
                        default= '{"n_estimators": 100}',
                        type=str,
                        required=False,
                        help='String in dictionary form, specifying the number of estimators.')

    parser.add_argument('-show_roc',
                        default=False,
                        type=bool,
                        help="Plot the roc curve for the model.")

    CONFIG = parser.parse_args()
    CONFIG.model_args = ast.literal_eval(CONFIG.model_args)

    main(CONFIG)
    #logger.debug('\n' + '='*50)
