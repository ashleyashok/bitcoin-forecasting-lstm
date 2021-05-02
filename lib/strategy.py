import numpy as np
import backtrader as bt

def create_trading_strategy(predicted):
    '''
    Parameters
    ----------
    predicted : LIST (FLOAT)
        Contains the list of predicted values by the model

    Returns
    -------
    signal : LIST(BOOLEAN)
        Trading strategy where 1 represents buy and 0 represents sell

    '''
    signal = np.where(predicted > 0, 1, 0)
    
    return signal


class AISignal(bt.SignalStrategy):
    '''
        Class to build backtrader trading strategy based on model's output
    '''

    def log(self, txt, dt=None):
    
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
    
        # Keep a reference to the "close" line in the data[0] dataseries
    
        self.dataclose = self.datas[0].close
    
        self.signal = self.datas[0].signal
    
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
    
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                      order.executed.value,
                      order.executed.comm))
    
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                          (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
    
            self.bar_executed = len(self)
    
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
    
        # self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
    
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                  (trade.pnl, trade.pnlcomm))
        
    def next(self):
        self.log(' Close, %.2f' % self.dataclose[0])
                
        #Check if we are in the market 
        if not self.position :
            
            if self.signal[0] == 1:
    
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
    
                #Keep track of the created order to avoid a 2nd order
                # self.order = self.buy(exectype=bt.Order.Close)
                self.order = self.buy()
                
        else:
            
            #Already in the market...we might sell
                       
            if self.signal[0] == 0:
    
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
    
                # self.order = self.sell(exectype=bt.Order.Close)
                self.order = self.sell()
    