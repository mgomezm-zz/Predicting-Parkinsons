import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize,scale

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,ExtraTreesClassifier


from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc

import tsne

def plot_ROC(probas, y_true, classes):
    '''
    Given prediction probabilities, and true classes, plot an ROC curve
    '''
    fpr, tpr, thresholds = roc_curve(y_true, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)
    #Plot ROC curve
    plt.plot(fpr, tpr,linewidth=2.0, label='AUC = %0.2f, %s' %(roc_auc, " vs ".join(classes.keys())))

def cross_validate(df):
    '''
    INPUT: pandas DataFrame, numpy ndarray, sklearn estimator
    OUTPUT: N/A

    Creates train-test split, fits model and then shows performance report.
    '''
    plt.figure()
    # models = []
    # for data in [df[(df.label == "PD") | (df.label == "SWEDD")],
    #                 df[(df.label == "PD") | (df.label == "HC")],
    #                 df[(df.label == "HC") | (df.label == "SWEDD")]]:
    #     X = scale(data.drop("label", 1).values)
    #     #X = data.drop("label", 1).values
    #     classes = {l:i for i,l in enumerate(set(data.label))}
    #     y = np.array([classes[i] for i in data.label])
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    #     final_model = RandomForestClassifier(n_estimators=200,  compute_importances=True)
    #     #final_model = ExtraTreesClassifier(n_estimators=200,  compute_importances=True)
    #     #final_model = LogisticRegression()
    #     final_model.fit(X_train,y_train)
    #     print "done training model", classes
    #     y_pred_probas = final_model.predict_proba(X_test)
    #     plot_ROC(y_pred_probas, y_test, classes)
    #     print 100*final_model.feature_importances_
    #     print

    X = scale(df.drop("label", 1).values)
    #X = data.drop("label", 1).values
    classes = {"PD":1, "SWEDD":1, "HC":0}
    y = np.array([classes[i] for i in df.label])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    #final_model = RandomForestClassifier(n_estimators=200,  compute_importances=True)
    #final_model = ExtraTreesClassifier(n_estimators=200,  compute_importances=True)
    final_model = LogisticRegression()
    final_model.fit(X_train,y_train)
    print "done training model", classes
    y_pred_probas = final_model.predict_proba(X_test)
    plot_ROC(y_pred_probas, y_test, classes)
    #print 100*final_model.feature_importances_
    print final_model.coef_
    print
    print



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

    #make predictions
    # print model
    # print "Train score:", model.score(X_train, y_train)
    # print "Test score:", model.score(X_test, y_test)
    # print "Classification Report:", classification_report(y_pred, y_test)
    # print


    n_classifiers = len(classifiers)

    plt.figure(figsize=(3 * 2, n_classifiers * 2))
    plt.subplots_adjust(bottom=.2, top=.95)

    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X, y)

        y_pred = classifier.predict(X)
        classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
        print("classif_rate for %s : %f " % (name, classif_rate))

        # View probabilities=
        xx = np.linspace(3, 9, 100)
        yy = np.linspace(1, 5, 100).T
        xx, yy = np.meshgrid(xx, yy)
        Xfull = np.c_[xx.ravel(), yy.ravel()]
        probas = classifier.predict_proba(Xfull)
        n_classes = np.unique(y_pred).size
        for k in range(n_classes):
            plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
            plt.title("Class %d" % k)
            if k == 0:
                plt.ylabel(name)
            imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                       extent=(3, 9, 1, 5), origin='lower')
            plt.xticks(())
            plt.yticks(())
            idx = (y_pred == k)
            if idx.any():
                plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='k')

    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')





if __name__ == "__main__":
    df_status = pd.read_csv("./data/Patient_Status.csv")[['PATNO', 'ENROLL_CAT']]
    patient_class = dict(df_status.dropna().values)

    df_DAT = pd.read_csv("./data/DaTscan_Striatal_Binding_Ratio_Results.csv")
    df_DAT = df_DAT[df_DAT.EVENT_ID == "SC"]
    df_DAT = df_DAT.drop("EVENT_ID", 1)
    #df_DAT['label'] = df_DAT.PATNO.apply(lambda x: patient_class.get(x, np.nan))

    ## Non-motor assessment
    df_smell = pd.read_csv("./data/Univ._of_Pennsylvania_Smell_ID_Test.csv")
    #df_smell = df_smell[df_smell.EVENT_ID == "SC"]
    df_smell = df_smell[['PATNO',"UPSITBK1","UPSITBK2","UPSITBK3","UPSITBK4"]]
    #df_smell['label'] = df_smell.PATNO.apply(lambda x: patient_class.get(x, np.nan))

    df_cognitive = pd.read_csv("./data/Montreal_Cognitive_Assessment__MoCA_.csv")
    df_cognitive = df_cognitive[df_cognitive.EVENT_ID == "SC"]
    df_cognitive = df_cognitive[["PATNO","MCATOT"]]
    # #df_cognitive['label'] = df_cognitive.PATNO.apply(lambda x: patient_class.get(x, np.nan))

    df_semantic = pd.read_csv("./data/Semantic_Fluency.csv")
    df_semantic = df_semantic[df_semantic.EVENT_ID == "BL"]
    df_semantic = df_semantic[["PATNO","VLTANIM","VLTVEG","VLTFRUIT"]]

    df_symbol = pd.read_csv("./data/Symbol_Digit_Modalities_Text.csv")
    df_symbol = df_symbol[df_symbol.EVENT_ID == "BL"]
    df_symbol= df_symbol[["PATNO","SDMTOTAL"]]

    df_learning = pd.read_csv("./data/Hopkins_Verbal_Learning_Test.csv")
    df_learning = df_learning[df_learning.EVENT_ID == "BL"]
    df_learning = df_learning[["PATNO","HVLTRT1","HVLTRT2","HVLTRT3","HVLTRDLY","HVLTREC","HVLTFPRL","HVLTFPUN","HVLTVRSN"]]

    df_sleep = pd.read_csv("./data/Epworth_Sleepiness_Scale.csv")
    df_sleep = df_sleep[df_sleep.EVENT_ID == "BL"]
    df_sleep = df_sleep[["PATNO","ESS1","ESS2","ESS3","ESS4","ESS5","ESS6","ESS7","ESS8"]]

    df_numseq = pd.read_csv("./data/Letter-Number_Sequencing__PD_.csv")
    df_numseq = df_numseq[df_numseq.EVENT_ID == "BL"]
    df_numseq = df_numseq[["PATNO","LNS_TOTRAW"]]

    df_line = pd.read_csv("./data/Benton_Judgment_of_Line_Orientation.csv")
    df_line = df_line[df_line.EVENT_ID == "BL"]
    df_line = df_line[["PATNO","JLO_TOTRAW"]]

    df_rem = pd.read_csv("./data/REM_Sleep_Disorder_Questionnaire.csv")
    df_rem = df_rem[df_rem.EVENT_ID == "BL"]
    df_rem = df_rem[["PATNO","PTCGBOTH","DRMVIVID","DRMAGRAC","DRMNOCTB","SLPLMBMV","SLPINJUR","DRMVERBL","DRMFIGHT","DRMUMV","DRMOBJFL","MVAWAKEN","DRMREMEM","SLPDSTRB","STROKE","HETRA","PARKISM","RLS","NARCLPSY","DEPRS","EPILEPSY","BRNINFM","CNSOTH"]]

    ## medical
    df_vital = pd.read_csv("./data/Vital_Signs.csv")
    df_vital = df_vital[df_vital.EVENT_ID == "BL"]
    df_vital = df_vital[["PATNO","WGTKG","HTCM","TEMPC","BPARM","SYSSUP","DIASUP","HRSUP","SYSSTND","DIASTND","HRSTND"]]


    ## depression
    df_depression = pd.read_csv("./data/Geriatric_Depression_Scale__Short_.csv")
    df_depression = df_depression[df_depression.EVENT_ID == "BL"]
    df_depression = df_depression[["PATNO","GDSSATIS","GDSDROPD","GDSEMPTY","GDSBORED","GDSGSPIR","GDSAFRAD","GDSHAPPY","GDSHLPLS","GDSHOME","GDSMEMRY","GDSALIVE","GDSWRTLS","GDSENRGY","GDSHOPLS","GDSBETER"]]

    ## protein
    df_protein = pd.read_csv("./ayasdi_csf_proteins.csv")

    #motor assesments
    df_UPDRSI = pd.read_csv("./data/MDS_UPDRS_Part_I__Patient_Questionnaire.csv")
    df_UPDRSI = df_UPDRSI[df_UPDRSI.EVENT_ID == "BL"]
    #df_UPDRSI = df_UPDRSI[["PATNO","NUPSOURC","NP1SLPN","NP1SLPD","NP1PAIN","NP1URIN","NP1CNST","NP1LTHD","NP1FATG"]]
    df_UPDRSI = df_UPDRSI[["PATNO","NP1SLPN","NP1SLPD","NP1PAIN","NP1URIN","NP1CNST","NP1LTHD","NP1FATG"]]

    df_UPDRSII = pd.read_csv("./data/MDS_UPDRS_Part_II__Patient_Questionnaire.csv")
    df_UPDRSII = df_UPDRSII[df_UPDRSII.EVENT_ID == "BL"]
    #df_UPDRSII = df_UPDRSII[["PATNO","NUPSOURC","NP2SPCH","NP2SALV","NP2SWAL","NP2EAT","NP2DRES","NP2HYGN","NP2HWRT","NP2HOBB","NP2TURN","NP2TRMR","NP2RISE","NP2WALK","NP2FREZ"]]
    df_UPDRSII = df_UPDRSII[["PATNO","NP2SPCH","NP2SALV","NP2SWAL","NP2EAT","NP2DRES","NP2HYGN","NP2HWRT","NP2HOBB","NP2TURN","NP2TRMR","NP2RISE","NP2WALK","NP2FREZ"]]


    #df = df_DAT[['PATNO','CAUDATE_R','CAUDATE_L','PUTAMEN_R','PUTAMEN_L']]
    #df = pd.merge(pd.merge(df_DAT, df_status, on='PATNO'), df_smell, on='PATNO')
    #df = df.dropna(subset=['ENROLL_CAT'])

    #df = pd.merge(df_DAT, df_smell, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df_smell, df_cognitive, on = ['PATNO'], how = 'outer')

    #df = pd.merge(df, df_vital, on = ['PATNO'], how = 'outer')

    #df = df_DAT
    #df = df_rem
    #df = df_smell

    #df = df_sleep
    #df = df_learning
    #df = df_cognitive # x
    #df = df_line # x
    #df = df_numseq # x
    #df = df_symbol
    #df = df_semantic # x

    #df= df_protein
    #df= df_depression

    #df = pd.merge(df_smell, df_UPDRSII, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_depression, on = ['PATNO'], how = 'outer')

    #df = df_UPDRSI
    df = df_UPDRSII

    #df = pd.merge(df_UPDRSI, df_UPDRSII, on = ['PATNO'], how = 'outer')

    #df = pd.merge(df_rem, df_smell, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_UPDRSI, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_UPDRSII, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_sleep, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_learning, on = ['PATNO'], how = 'outer') # x
    #df = pd.merge(df, df_symbol, on = ['PATNO'], how = 'outer') # x
    #df = pd.merge(df, df_depression, on = ['PATNO'], how = 'outer')

    #df = pd.merge(df, df_DAT, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_cognitive, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_learning, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_sleep, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_semantic, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_symbol, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_numseq, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_line, on = ['PATNO'], how = 'outer')
    #df = pd.merge(df, df_depression, on = ['PATNO'], how = 'outer')

    df['label'] = df.PATNO.apply(lambda x: patient_class.get(x, np.nan))
    df = df.dropna() #subset=['label'])
    df = df.drop("PATNO",1)

    #X = scale(df.drop("label", 1).values)
    ##cross_validate(df[["NP2SPCH","NP2HWRT","NP2TRMR", "label"]])
    #cross_validate(df)


    x1 = df.NP2SPCH[df.label == "PD"].values
    y1 = df.NP2TRMR[df.label == "PD"].values
    x2 = df.NP2SPCH[df.label == "HC"].values
    y2 = df.NP2TRMR[df.label == "HC"].values
