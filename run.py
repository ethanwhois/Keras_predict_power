import os
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from core.data_processor import DataLoader
from core.model import Model
from keras.utils import plot_model

#plot results
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data,label='Prediction')
    plt.legend()
    plt.savefig('./output/results_1.png')

def plot_results_multiple(predicted_data, true_data,prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.legend()
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i*prediction_len)]
        plt.plot(padding+data,label='Prediction')
    plt.savefig('./output/results_multiple_1.png')

def main():
    #load parameters
    configs = json.load(open('./data/config.json','r'))
    if not os.path.exists(configs['model']['save_dir']):os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data',configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns'],

    )
    #create RNN model
    model=Model()
    model.build_model(configs)

    #loading trainning data
    x,y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    print(x.shape)
    print(y.shape)

    #training model
    model.train(
        x,
        y,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        save_dir=configs['model']['save_dir']
    )

    #test results
    x_test, y_test = data.get_test_data(
        seq_len= configs['data']['sequence_length'],
        normalise=configs['data']['normalise'],
    )

    #results visualization
    predictions_multiseq = model.predict_sequences_multiple(x_test,configs['data']['sequence_length'],configs['data']['sequence_length'])
    predictions_pointbypoint=model.predict_point_by_point(x_test)

    plot_results_multiple(predictions_multiseq,y_test,configs['data']['sequence_length'])
    plot_results(predictions_pointbypoint,y_test)

if __name__ == '__main__':
    main()