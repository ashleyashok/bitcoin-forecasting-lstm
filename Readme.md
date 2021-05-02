# Bitcoin Forecasting using AI and Backtrader Testing

## Description

The script contains different neural network model architectures which are represented by the constants

    RNN : Recurrent Neural Network
    SEQ2SEQ_LSTM: Sequence to Sequence LSTM Model
    CONV_GRU: Convolutional Network + GRU
    CONV_LSTM : Convolutional Network + LSTM

The predicted output from the model on test data is converted into a binary buy/sell hold strategy (No shorting)

Backtrader is used to implement that strategy against historical test data with a starting deposit of $10000 to test account balance growth over time for different model configurations 


## Pre-requisites

Install the libraries required from to run main.py from requirements.txt using the command

pip install -r requirements.txt