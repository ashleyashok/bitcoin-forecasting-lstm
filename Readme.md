# Bitcoin Forecasting using AI and Backtrader Testing

## About the project

The script contains different neural network model architectures which are represented by the constants

    RNN : Recurrent Neural Network
    SEQ2SEQ_LSTM: Sequence to Sequence LSTM Model
    CONV_GRU: Convolutional Network + GRU
    CONV_LSTM : Convolutional Network + LSTM

Historical bitcoin prices are used to train the model along with various bitcoin specific and macro economic features to predict a buy/sell binary indicator. The predicted output from the model on test data is converted into a trading strategy (No shorting)

Backtrader is used to implement that strategy against historical test data with a starting deposit of $10000 to test account balance growth over time for different neural network model configurations.

A sample output from Backtrader testing built on the strategy developed by Convolutional + GRU based model is as follows:

![Sample AI model Backtrader testing results](/figures/CONV_GRU_Backtrader_Results.png)

## Pre-requisites

Install the libraries required from to run "main.py" from requirements.txt using the command

pip install -r requirements.txt


