import random 
import numpy as np
import pandas as pd 
import keras
import tensorflow as tf
from datetime import datetime, timedelta


#for testing
def model_random(df, t_pred):  
    prob_open = random.random()
    return [random.choice([0, 1]), prob_open]


#manually set to known dates
def model_ground_truth(df, t_pred):
  
    t_open = pd.to_datetime('2022-07-16 18:00:00')
    t_clos = pd.to_datetime('2022-08-26 00:00:00')

    if t_pred <= t_open or t_pred >= t_clos: status = 0
    else: status = 1
    
    return status


#used in model_predict_open_hatch - ignore this
def test(true, pred):
        return 0

#neural network model
def model_predict_open_hatch(df:pd.core.frame.DataFrame, t_pred:datetime):
    #get overall info
    mean = df["pressure_osi"].mean()
    std = df["pressure_osi"].std()
    #get local info
    dfLocal = df[(df.timestamp > t_pred-timedelta(days=2))&
                            (df.timestamp < t_pred)]
    localMean = dfLocal["pressure_osi"].mean()
    localSTD = dfLocal["pressure_osi"].std()
    #build test input
    testInput = tf.constant([[mean, std, localMean, localSTD]])

    #load model from 'models/model'
    model = keras.models.load_model("models/model", custom_objects={"test":test})
    #predict
    prediction = model.predict(testInput)[0][0]
    #rectify data that is NaN
    if np.isnan(prediction):
        prediction = 0.499999

    return [int(prediction+0.5), 2*abs(0.5-prediction)]