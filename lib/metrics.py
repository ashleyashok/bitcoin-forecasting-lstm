import pandas as pd
import numpy as np

def compute_returns(df, price_col):
    '''
    Assumes that the signal is for that day i.e. if a signal of 
    1 exists on the 12th of January, I should buy before that day begins
    '''
    new_df = df.copy()
    
    new_df['mkt_returns'] = new_df[price_col].pct_change()
    new_df['system_returns'] = new_df['mkt_returns']*new_df['signal']
    
    new_df['system_equity'] = np.cumprod(1+new_df.system_returns) - 1
    new_df['mkt_equity'] = np.cumprod(1+new_df.mkt_returns) - 1
    
    return new_df

def compute_metrics(df):
    new_df = df.copy()
    
    new_df['system_equity']=np.cumprod(1+new_df.system_returns) -1
    system_cagr=(1+new_df.system_equity.tail(n=1))**(252/new_df.shape[0])-1
    new_df.system_returns= np.log(new_df.system_returns+1)
    system_sharpe = np.sqrt(252)*np.mean(new_df.system_returns)/np.std(new_df.system_returns)

    new_df['mkt_equity']=np.cumprod(1+new_df.mkt_returns) -1
    mkt_cagr=(1+new_df.mkt_equity.tail(n=1))**(252/new_df.shape[0])-1
    new_df.mkt_returns= np.log(new_df.mkt_returns+1)
    mkt_sharpe = np.sqrt(252)*np.mean(new_df.mkt_returns)/np.std(new_df.mkt_returns)
    
    system_metrics = (system_cagr, system_sharpe)
    market_metrics = (mkt_cagr, mkt_sharpe)
    
    return system_metrics, market_metrics
