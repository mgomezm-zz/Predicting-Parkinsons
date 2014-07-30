Predicting-Parkinson's
=====================

[Project Link](http://www.mgomezm.com/)


[Zipfian](http://www.zipfianacademy.com/) and [Bayes Impact](http://www.bayesimpact.org/) project. Using PPMI data funded by the Michale J. Fox Foundation.

## GOAL
Improve on a well known method to diagnose Parkinson’s. More specifically using machine learning find predictor variables that can classify against a subtype of Parkinson's (SWEDD).

## PROJECT DESCRIPTION
A widely used method to diagnose Parkinson's involves an expensive SPECT brain DaTSCAN. This procedure basically measures dopamine level. Out of the patients who have been diagnosed with early Parkinson's there exist a group called Scans Without Evidence of Dopaminergic Deficit (SWEDD). These patients experience symptoms of Parkinson's but fail to show a low level of dopamine. This means that using a DaTSCAN to identity SWEDD patients is not possible. Therefore, I decided to explore the PPMI database and possibly find predictors that can identify SWEDD subjects.

#### steps·
* Background Research
* ETL, on massive horizontal data.·
* Data directories(each directories had about 4 data files and each file on average had about 50 variables):
    * DaTSCAN, MRI measurements·
    * FMRI, MRI, SPECT Image Data
    * Bio-Specimen Reconciliation
    * Neurological Exam
    * Medical History
    * Subject Characteristics
    * Clinical Assessments·
    * FMRI, MRI, SPECT Image Data
    * Subject Characteristics
* EDA, logistic regression on DaTSCAN features. Obviously not good at classifying SWEDD vs control (HC).
* More EDA, found out that clinical assessments are good predictors of Parkinson's.
* Using model selection found out REM, SMELL, and  MOTOR assessments together can identify SWEDD.
* Create a web app that is more accessible and allows for quick evaluation of PD based on three questions from the MOTOR assessment.

## Conclusion
Using machine learning and various techniques for data analysis found a cheap, non-invasive method to classify Parkinson's. In fact this model outperforms using DaTSCAN. Based on this outcome created a web app for early detection of Parkinson's.

#### generate_simpe_model.py
The Motor assessment has thirteen questions. To attract users, the model has been simplified to three questions. Using variable importance selected the most import features. Below is the CLI interface to this model generator. You can check model performance, plot ROC curves, run multi-class classification, and save your model. After trying out out different machine learning models found out that Random Forest had a higher ROC than Logistic, but no significant improvements over Extremely Randomized Trees or Gradient Tree Boosting.

Usage:
---------------

    usage: generate_simple_model.py [-h] [-patient_file PATIENT_FILE]
                                    [-data_file DATA_FILE]
                                    [-multiclass MULTICLASS]
                                    [-model_type {Logistic,RandomForest,ExtraTrees,GradBoost}]
                                    [-serialize_model SERIALIZE_MODEL]
                                    [-model_args MODEL_ARGS] [-show_roc SHOW_ROC]

    Train different models, show ROC curves, and serialize the chosen model.

    optional arguments:
    -h, --help            show this help message and exit
    -patient_file PATIENT_FILE
                            CSV with patient info (labels). (default:
                            ./../data/Patient_Status.csv)
    -data_file DATA_FILE  CSV with training data. (default: ./../data/MDS_UPDRS_
                            Part_II__Patient_Questionnaire.csv)
    -multiclass MULTICLASS
                            Three classes available (HC, SWEDD, PD). Generate
                            model for all class (all-pairs)? or merge SWEDD and PD
                            together for binary classifcatoin. (default: False)
    -model_type {Logistic,RandomForest,ExtraTrees,GradBoost}
                            Type of model to fit. (default: Logistic)
    -serialize_model SERIALIZE_MODEL
                            Save model. (default: False)
    -model_args MODEL_ARGS
                            String in dictionary form, specifying the number of
                            estimators. (default: {"n_estimators": 100})
    -show_roc SHOW_ROC    Plot the roc curve for the model. (default: False)

