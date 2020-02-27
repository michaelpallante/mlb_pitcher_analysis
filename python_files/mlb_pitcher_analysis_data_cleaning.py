#!/usr/bin/env python
# coding: utf-8

# # MLB Pitcher Analysis
# 
# ## Data Cleaning

# In[413]:


#Import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")
pd.options.display.max_columns = 100


# In[372]:


#Import ibp_pitcher data and create pitcher dataframe
pitcher_df = pd.read_csv('../data/ibp_pitcher.csv')

#Overview of pitcher dataframe
# pitcher_df


# In[373]:


#Create fix_date function to fix bs_count column, as there are some ball-strike counts that are appearing as dates
# and need to be formatted correctly
def fix_date(x):
    if '-' in x:
        return x
    else:
        return ('-').join(x.split('/')[0:-1])
    
#apply fix_date function to the bs_count column to clean/correct all values
pitcher_df['bs_count'] = pitcher_df['bs_count'].map(fix_date)


# In[374]:


#replace '#VALUE!' values in game_state column with NaN values, as these values are unknown and missing
pitcher_df['game_state'] = pitcher_df['game_state'].replace('#VALUE!', np.nan)


# In[375]:


#Split bs_count column into 2 separate columns: ball_count and strike_count
#ball_count is the number of balls in the count, between 0-3 balls
#strike_count is the number of strikes in the count, between 0-2 strikes
pitcher_df[['ball_count', 'strike_count']] = pitcher_df.bs_count.str.split("-",expand=True)

#Split game_state column into 2 separate columns: baserunner_count and out_count
#baserunner_count is the total number of opposing players on base, between 0-3 players
#out_count is the total number of outs, between 0-2 outs
pitcher_df[['baserunner_count', 'out_count', 'col_to_be_removed']] = pitcher_df.game_state.str.split(":",expand=True)

#created the baserunner_on_first column to indicate if there is a baserunner on first base
pitcher_df['baserunner_on_first'] = pitcher_df['baserunner_count'].astype(str).map({'---': 0, '1--': 1, '-2-': 0,
                                                                                    '--3': 0, '12-': 1, '1-3': 1,
                                                                                    '-23': 0, '123': 1})
#created the baserunner_on_second column to indicate if there is a baserunner on second base
pitcher_df['baserunner_on_second'] = pitcher_df['baserunner_count'].astype(str).map({'---': 0, '1--': 0, '-2-': 1,
                                                                                    '--3': 0, '12-': 1, '1-3': 0,
                                                                                    '-23': 1, '123': 1})
#created the baserunner_on_third column to indicate if there is a baserunner on third base
pitcher_df['baserunner_on_third'] = pitcher_df['baserunner_count'].astype(str).map({'---': 0, '1--': 0, '-2-': 0,
                                                                                    '--3': 1, '12-': 0, '1-3': 1,
                                                                                    '-23': 1, '123': 1})

#created individual variables for whether a pitch resulted in a ball or strike called
pitcher_df['result_ball'] = pitcher_df['pitch_result'].astype(str).map({'ball': 1, 'called_strike': 0})
pitcher_df['result_strike'] = pitcher_df['pitch_result'].astype(str).map({'ball': 0, 'called_strike': 1})

#created individual variables for all six different pitch types
pitcher_df['pitch_type_CB'] = pitcher_df['pitch_type'].astype(str).map({'CB': 1, 'CH': 0, 'CT': 0, 'FF': 0,
                                                                        'FT': 0, 'SL': 0})

pitcher_df['pitch_type_CH'] = pitcher_df['pitch_type'].astype(str).map({'CB': 0, 'CH': 1, 'CT': 0, 'FF': 0,
                                                                        'FT': 0, 'SL': 0})

pitcher_df['pitch_type_CT'] = pitcher_df['pitch_type'].astype(str).map({'CB': 0, 'CH': 0, 'CT': 1, 'FF': 0,
                                                                        'FT': 0, 'SL': 0})

pitcher_df['pitch_type_FF'] = pitcher_df['pitch_type'].astype(str).map({'CB': 0, 'CH': 0, 'CT': 0, 'FF': 1,
                                                                        'FT': 0, 'SL': 0})

pitcher_df['pitch_type_FT'] = pitcher_df['pitch_type'].astype(str).map({'CB': 0, 'CH': 0, 'CT': 0, 'FF': 0,
                                                                        'FT': 1, 'SL': 0})

pitcher_df['pitch_type_SL'] = pitcher_df['pitch_type'].astype(str).map({'CB': 0, 'CH': 0, 'CT': 0, 'FF': 0,
                                                                        'FT': 0, 'SL': 1})


# In[376]:


#fixing the baserunner_count observations to reflect whether there are between 0-3 baserunners
pitcher_df['baserunner_count'] = pitcher_df['baserunner_count'].str.replace('---','0').str.replace('1--','1')
pitcher_df['baserunner_count'] = pitcher_df['baserunner_count'].str.replace('-2-','1').str.replace('--3','1')
pitcher_df['baserunner_count'] = pitcher_df['baserunner_count'].str.replace('12-','2').str.replace('1-3','2')
pitcher_df['baserunner_count'] = pitcher_df['baserunner_count'].str.replace('-23','2').str.replace('123','3')

#fixing the out_count observations to reflect whether there are between 0-2 outs
pitcher_df['out_count'] = pitcher_df['out_count'].str.replace('00','0').str.replace('01','1').str.replace('02','2')

#changing the ball_count, strike_count, baserunner_count, and out_count types to integers
pitcher_df['ball_count'] = pitcher_df['ball_count'].astype(int)
pitcher_df['strike_count'] = pitcher_df['strike_count'].astype(int)

#dropping the bs_count, game_state, and col_to_be_removed columns, as they are no longer needed in our analysis
pitcher_df = pitcher_df.drop(['bs_count', 'game_state', 'col_to_be_removed'], axis = 1)


# In[377]:


#overview of all of our variables in the pitcher dataframe
# pitcher_df.info()


# In[378]:


#create dataframe of pitcher 1's data only
pitcher1 = pitcher_df[pitcher_df.pitcherid == 1]

#create dataframe of pitcher 2's data only
pitcher2 = pitcher_df[pitcher_df.pitcherid == 2]

#create dataframe of pitcher 3's data only
pitcher3 = pitcher_df[pitcher_df.pitcherid == 3]


# In[379]:


# check the number of missing (NAN) values in each pitcher dataframe
# print(pitcher1.isna().sum())
# print(pitcher2.isna().sum())
# print(pitcher3.isna().sum())


# In[380]:


#create separate dataframes of pitcher 1's data for before and after the all star break
pitcher1_before = pitcher1[pitcher1.all_star == 'before']
pitcher1_after = pitcher1[pitcher1.all_star == 'after']

#create separate dataframes of pitcher 2's data for before and after the all star break
pitcher2_before = pitcher2[pitcher2.all_star == 'before']
pitcher2_after = pitcher2[pitcher2.all_star == 'after']

#create separate dataframes of pitcher 3's data for before and after the all star break
pitcher3_before = pitcher3[pitcher3.all_star == 'before']
pitcher3_after = pitcher3[pitcher3.all_star == 'after']

