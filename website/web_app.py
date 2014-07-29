import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy.random import choice
from itertools import combinations

from flask import Flask
from flask import request
from flask import render_template
from flask import redirect, url_for
import json

import cPickle as pickle

app = Flask(__name__)

@app.route('/')
def home():
    #return render_template("my-form1.html")
    return render_template("index.html")

@app.route('/results')
def results():
    if request.args:
        answers = request.args['submit_val'].split('/t')
        names_values = {n:float(v) for n,v in zip(col_names, answers)}
        print "received input"
    else:
        names_values = {n:0 for n in col_names}
        print "empty"

    print names_values

    #if the user submitted remove it 
    for d in dout:
        if any(d_i["key"] == "YOU" for d_i in d):
            d.pop()

    #add user
    for d, vu in zip(dout, combinations(col_names, 2)):
        d.append({"key":"YOU","values":get_locations([names_values[vu[1]]], [names_values[vu[0]]])})

    #classify current user, parkinson prob
    prob = final_model.predict_proba(map(lambda x: (x*4)/10., names_values.values())).squeeze()[1]
    print "probability = ", prob
    return render_template('results.html', likelihood = prob)#, prediction=prediction)

@app.route("/data")
def data():
    return json.dumps(dout)

def get_locations(col1, col2, jitter = False):
    if jitter:
        return [{"x":x*(10./4)+.4*np.random.randn()*jitter, \
                    "y":y*(10./4)+.4*np.random.randn()*jitter} \
                    for x,y in zip(col1, col2)]
    else:
        return [{"x":x, "y":y } for x,y in zip(col1, col2)]

if __name__ == '__main__':
    global final_model
    global col_names
    global dout

    col_names = (u'NP2SPCH', u'NP2HWRT', u'NP2TRMR')
    #loading data
    N = 150
    dout = []
    ax_val = {'NP2SPCH':'Speech', 'NP2HWRT':'Hand Writing', 'NP2TRMR':'Tremor'}
    df = pd.read_csv("./../data/ppmi.csv")
    for v,u in combinations(col_names,2):
        xcol1 = choice(df[(df.label == "PD") | (df.label == "SWEDD")][[u]].values.squeeze(), N)
        ycol2 = choice(df[(df.label == "PD") | (df.label == "SWEDD")][[v]].values.squeeze(), N)
        d1 = {"key":"PD", "values": get_locations(xcol1, ycol2, True), "axis_names":[ax_val[u], ax_val[v]]}

        xcol1 = choice(df[df.label == "HC"][[u]].values.squeeze(), N)
        ycol2 = choice(df[df.label == "HC"][[v]].values.squeeze(), N)
        d2 = {"key":"HC", "values": get_locations(xcol1, ycol2, True), "axis_names":[ax_val[u], ax_val[v]]}
        dout.append([d1, d2])

    # unpickle model
    with open('./../model/Parkinson_two_class.model', 'rb') as fr:
        final_model = pickle.loads(fr.read())

    #app.debug = True
    app.run(host="0.0.0.0", port=80)
