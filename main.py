"""
The script contains different neural network model architectures which are represented by the constants

    RNN : Recurrent Neural Network
    SEQ2SEQ_LSTM: Sequence to Sequence LSTM Model
    CONV_GRU: Convolutional Network + GRU
    CONV_LSTM : Convolutional Network + LSTM

The predicted output from the model on test data is converted into a binary buy/sell hold strategy (No shorting)

Backtrader is used to implement that strategy against historical test data with a starting deposit of $10000 to test account balance growth over time for different model configurations 

"""


import os,sys
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.vis_utils import plot_model
import backtrader as bt
from lib.dataloader import DataLoader
from lib.metrics import compute_returns, compute_metrics
from lib.strategy import create_trading_strategy, AISignal


def plot_training_curves(history):
    ''' Plot the train and validation loss curves'''
    
    train_loss_values = history.history["loss"] #training loss
    val_loss_values = history.history["val_loss"] #validation loss
    plt.clf()  
    plt.plot(train_loss_values, label="Train Loss")
    plt.plot(val_loss_values, label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curves")
    plt.show()
    return

def plot_results(predicted_data, true_data):
    ''' Plot true data vs prediction'''
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    return    


def concatenate_strat_to_test(test_df, trading_signal, seq_len):
    '''
    Concatenates the trading signal to the test_df
    '''
    new_df = test_df.copy()
    
    # Start and stop length. Start at the seq_len or lookback window - 1
    # This is because if the lookback window is set to 27, we are looking
    # at the last 26 and then predicting for the 27th
    new_signal = np.hstack(([np.nan], trading_signal))
    start = seq_len-2
    stop = start + len(new_signal)
    
    # Add the signal to the dataframe
    new_df = new_df.iloc[start:stop, :]
    new_df['signal'] = new_signal.reshape(-1,1)
    
    return new_df

def plot_returns(df):
    ''' Plotting system returns vs market returns'''
    
    df[['system_equity','mkt_equity']].plot()
    plt.show()
    return


def model_type(model_name):
    '''
    Parameters
    ----------
    model_name : STRING
      The model name required for the build
            RNN : Recurrent Neural Network
            SEQ2SEQ_LSTM: Sequence to Sequence LSTM Model
            CONV_GRU: Convolutional Network + GRU
            CONV_LSTM : Convolutional Network + LSTM


    Returns
    -------
    model : TENFORFLOW MODEL
        Keras Sequential model with built layers.

    '''
    
    if model_name=='SEQ2SEQ_LSTM':
        model = keras.models.Sequential([
                                            keras.layers.LSTM(100, return_sequences=True, input_shape=[20, 16], activation="tanh"),
                                            keras.layers.Dropout(0.05),
                                            keras.layers.LSTM(100, return_sequences=True, activation="tanh"),
                                            keras.layers.LSTM(100, return_sequences=False, activation="tanh"),
                                            keras.layers.Dropout(0.05),
                                            keras.layers.Dense(1, activation='linear')
                                        ])
    elif model_name == 'RNN':
        model = keras.models.Sequential([
                                            keras.layers.SimpleRNN(100, return_sequences=True, input_shape=[20, 16], activation="tanh"),
                                            keras.layers.Dropout(0.05),
                                            keras.layers.SimpleRNN(100, return_sequences=True, activation="tanh"),
                                            keras.layers.SimpleRNN(100, return_sequences=False, activation="tanh"),
                                            keras.layers.Dropout(0.05),
                                            keras.layers.Dense(1, activation='linear')
                                        ])

    elif model_name == 'CONV_LSTM':
        model = keras.models.Sequential([
                                            keras.layers.Conv1D(filters=64, kernel_size=5,
                                                                strides=1, padding="causal",
                                                                activation="relu",
                                                                input_shape=[20, 16]),
                                            keras.layers.Dropout(0.05),
                                            keras.layers.LSTM(100, return_sequences=True,activation="tanh"),
                                            keras.layers.Dropout(0.05),
                                            keras.layers.LSTM(100, return_sequences=False,activation="tanh"),
                                            keras.layers.Dropout(0.05),
                                            keras.layers.Dense(1, activation='linear')
                                        ])   
        
    elif model_name =='CONV_GRU':
        model = keras.models.Sequential([
                                            keras.layers.Conv1D(filters=64, kernel_size=5,
                                                                strides=1, padding="causal",
                                                                activation="relu",
                                                                input_shape=[20, 16]),
                                            keras.layers.Dropout(0.05),
                                            keras.layers.GRU(100, return_sequences=True,activation="tanh"),
                                            keras.layers.Dropout(0.05),
                                            keras.layers.GRU(100, return_sequences=False,activation="tanh"),
                                            keras.layers.Dropout(0.05),
                                            keras.layers.Dense(1, activation='linear')
                                        ])   

       
    return model

if __name__ == '__main__':
    
    # Define Constants
    MODEL_NAME = 'CONV_GRU'
    TRAIN_TEST_SPLIT =  0.8
    FILENAME = 'btc_dataset.csv'
    
    
    data_path = os.path.join(os.getcwd(), 'data', FILENAME) # get the path to the dataset
   
    df = pd.read_csv(data_path,infer_datetime_format=True, parse_dates=['Date'], index_col=['Date']) # Read dataset as a dataframe
    
      
    data = DataLoader(df,TRAIN_TEST_SPLIT) # Create an object to load train and test sets      
    
    # Extract X & y train test split with defined lookback window and normalization
    x_train, y_train = data.get_train_data(lookback_window=21,normalize=True)
    x_test, y_test = data.get_test_data(lookback_window=21,normalize=True)
    
    # If choose to load a pre-build model
    # model = keras.models.load_model('Saved Models\\'+ MODEL_NAME)
    
    # OR Create a new model based on the MODEL_NAME defined
    model = model_type(model_name=MODEL_NAME) # Instantiate the Sequential model with defined layers
    model.compile(loss='mse', optimizer='adam') # Compile the model
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),          
                EarlyStopping(monitor='val_loss', patience=10, verbose=1)] # Set Callbacks
    
    history = model.fit(x_train,y_train,epochs=40,batch_size=16,
                        validation_split=0.2,
                        callbacks=callbacks) # Fit the training data
    
    
    # Plot the loss curve
    plot_training_curves(history)
    
    predicted = model.predict(x_test) # Create predicted forecast
    predicted = np.reshape(predicted, (predicted.size,)) # Reshape the predicted forecast into 1D numpy array
    
    plot_results(predicted, y_test) # Plot predited values vs true values
    
    signal = create_trading_strategy(predicted) # Create the trading strategy based on predicted points
    
    # Create a new dataframe with test data and predicted signal
    new_df = concatenate_strat_to_test(data.test_df, signal, 21) 
    new_df = compute_returns(new_df, "Closing Price (USD)") # Compute System and market returns
    system_metrics, market_metrics = compute_metrics(new_df) # Compute system and market metrics (CAGR and Sharpe)
    
    # Print out metrics
    print(f'System CAGR: {system_metrics[0][0]*100:.1f}%')
    print(f"System Sharpe: {system_metrics[1]:.1f}")
    
    print(f"Market CAGR: {market_metrics[0][0]*100:.1f}%")
    print(f"Market Sharpe: {market_metrics[1]:.1f}")
    
    plot_returns(new_df) # Plot System vs market returns
    
    # If choose to save the model
    # model.save('Saved Models\\' + MODEL_NAME)
    
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='Figures\\' + MODEL_NAME + '.png')
    
    # %% =============================================================================
    # Test Strategy using backtrader
    # =============================================================================
                   
    cerebro = bt.Cerebro(cheat_on_open=True) # Instantiate Backtrader Cerebro
    cerebro.broker.set_coc(True) # Set cheat on close true

    
    # Create the signal dataframe for backtrader
    signal_df = new_df.reset_index()
    signal_df['Opening Price (USD)'] = signal_df['Closing Price (USD)'].values
    signal_df['High'] = signal_df['Closing Price (USD)'].values
    signal_df['Low'] = signal_df['Closing Price (USD)'].values
    signal_df.dropna(subset=['signal'], inplace=True)
    
    # Define Signal index for backtrader
    signal_ind = signal_df.columns.tolist().index('signal')
    close_ind = signal_df.columns.tolist().index('Closing Price (USD)')
    date_ind = signal_df.columns.tolist().index('Date')
    open_ind = signal_df.columns.tolist().index('Opening Price (USD)')
    high_ind = signal_df.columns.tolist().index('High')
    low_ind = signal_df.columns.tolist().index('Low')

    # Create a class by inheriting PandasData and adding custom our custom signal column
    class PandasData_Signal(bt.feeds.PandasData):
        lines = ('signal',)
        params = ( ('dtformat', ('%Y-%m-%d')),
              ('signal',signal_ind),)
        
    # Creating the cerebro data object with the signal dataframe
    cerebro_data = PandasData_Signal(dataname=signal_df, dtformat=('%Y-%m-%d'), datetime=date_ind,high=high_ind, low=low_ind, open=open_ind, close=close_ind, volume=-1, openinterest=-1, signal=signal_ind)
    cerebro.adddata(cerebro_data)
    
    cerebro.broker.setcash(10000) # Setting initial cash to begin trading
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    cerebro.addstrategy(AISignal) # Adding the custom trading buy/sell strategy logic to cerebro
    cerebro.run() # Run the backtest and print log
    print('Final Portfolio Value: {0:8.2f}'.format(cerebro.broker.getvalue()))
    
    cerebro.plot(iplot=False, volume=False) # generates the transaction plot
    
