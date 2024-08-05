import pandas as pd


def split80_20(df):
    '''Iterates over individual users, splitting data in time into about 
       80% train data and 20% test, then returns these dfs as: train, test. 
       
       Assumes passed in df has a user_id column to get samples from all id's.
       Takes some time to run depending on size of df.
       '''

    # initialize df's
    train = pd.DataFrame()
    test = pd.DataFrame()

    # iterating over all unique user id's
    for user in df['user_id'].unique():
        
        # getting a single device's data
        user_df = df[df['user_id']==user]

        # define indices for 80% of device's data (rounding down to nearest int)
        cutoff_idx = 80*len(user_df) // 100
        # sorting data by start time
        sorted_user = user_df.sort_values('start').copy()
        # dividing data into 80/20 by index
        temp_train = sorted_user.iloc[:cutoff_idx]
        temp_test = sorted_user.iloc[cutoff_idx:]
        
        # appending user data to correct df
        train = train.append(temp_train)
        test = test.append(temp_test)
    
    # return complete df's
    return train, test
