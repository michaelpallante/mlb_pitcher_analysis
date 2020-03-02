#!/usr/bin/env python
# coding: utf-8

# # MLB Pitcher Analysis
# 
# ## Data Cleaning

# First, we must import the primary libraries that will be used throughout our notebooks.

# In[1]:


#Import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")
pd.options.display.max_columns = 100


# Next, we must import the ibp_pitcher data, which we will be using and manipulating throughout this study.

# In[2]:


#Import ibp_pitcher data and create pitcher dataframe
pitcher_df = pd.read_csv('../../data/ibp_pitcher.csv')

#Overview of pitcher dataframe
# pitcher_df


# We renamed the ibp_pitcher data to pitcher_df. This pitcher data contains 4862 total observations of pitch data and 20 different variables that we will be working with.
# 
# Through further observation, we can see a few columns that have some formatting issues that we will need to fix to be able to use these columns for evaluation.
# 
# Below, we create the fix_date function to fix the bs_count column, as there are some ball-strike counts that are appearing as date and need to be formatted correctly.

# In[3]:


#Create fix_date function to fix bs_count column, as there are some ball-strike counts that are appearing as dates
# and need to be formatted correctly
def fix_date(x):
    if '-' in x:
        return x
    else:
        return ('-').join(x.split('/')[0:-1])
    
#apply fix_date function to the bs_count column to clean/correct all values
pitcher_df['bs_count'] = pitcher_df['bs_count'].map(fix_date)


# We applpied the corrected bs_count column to our original dataset.
# 
# Next, we observe that there are many values in the game_state column that are listed as "#VALUE!". We are unable to decipher what these values are supposed to be, so we classify them as missing data by changing them to NaN values. However, it seems that the remaining data within these rows is still too valuable to discard, so we decide not to remove this missing data just yet.

# In[4]:


#replace '#VALUE!' values in game_state column with NaN values, as these values are unknown and missing
pitcher_df['game_state'] = pitcher_df['game_state'].replace('#VALUE!', np.nan)


# To be able to use the data within the bs_count and game_state columns, we must separate the data within those columns into columns of their own and create some new variables.
# 
# With the game_state column, we also notice that there are some formatting issues that we need to address. However, these formatting issues will be naturally corrected when we split this column into multiple columns.
# 
# Also, there are some valuable data transformations that we can make with the pitch_result and pitch_type variables that can be useful in further evaluation.
# 
# We display these two processes below and will explain further after the following cell.

# In[5]:


#Split bs_count column into 2 separate columns: ball_count and strike_count
#ball_count is the number of balls in the count, between 0-3 balls
#strike_count is the number of strikes in the count, between 0-2 strikes
pitcher_df[['ball_count', 'strike_count']] = pitcher_df.bs_count.str.split("-",expand=True)

#changing the ball_count and strike_count types to integers
pitcher_df['ball_count'] = pitcher_df['ball_count'].astype(int)
pitcher_df['strike_count'] = pitcher_df['strike_count'].astype(int)

#Split game_state column into 2 separate columns: baserunner_count and out_count
#baserunner_count is the total number of opposing players on base, between 0-3 players
#out_count is the total number of outs, between 0-2 outs
pitcher_df[['baserunner_count', 'out_count', 'col_to_be_removed']] = pitcher_df.game_state.str.split(":",expand=True)

#fixing the baserunner_count observations to reflect whether there are between 0-3 baserunners
pitcher_df['baserunner_count'] = pitcher_df['baserunner_count'].str.replace('---','0').str.replace('1--','1')
pitcher_df['baserunner_count'] = pitcher_df['baserunner_count'].str.replace('-2-','1').str.replace('--3','1')
pitcher_df['baserunner_count'] = pitcher_df['baserunner_count'].str.replace('12-','2').str.replace('1-3','2')
pitcher_df['baserunner_count'] = pitcher_df['baserunner_count'].str.replace('-23','2').str.replace('123','3')

#fixing the out_count observations to reflect whether there are between 0-2 outs
pitcher_df['out_count'] = pitcher_df['out_count'].str.replace('00','0').str.replace('01','1').str.replace('02','2')

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

#dropping the bs_count, game_state, and col_to_be_removed columns, as they are no longer needed in our analysis
pitcher_df = pitcher_df.drop(['bs_count', 'game_state', 'col_to_be_removed'], axis = 1)

#create .csv file of pitcher_df and save it to the data folder
# pitcher_df.to_csv('../data/pitcher_df.csv')


# We were able to separate the bs_count column into two new columns: ball_count and strike_count. Ball_count is the number of balls in the count, between 0-3 balls. Strike_count is the number of strikes in the count, between 0-2 strikes. We also needed to change ball_count and strike_count values to integers.
# 
# We were able to separate the game_state column into three new columns: baserunner_count, out_count, and col_to_be_removed. Baserunner_count is the total number of baserunners on base prior to the pitch. Out_count is the total number of outs prior to the pitch. Col_to_be_removed is an extra column that was created due to the formatting issues of some of the game_state cells. There is no useful information in this specific column and we will be removing that from the dataset. We also fixed the formatting issues in the baserunner_count and out_count columns.
# 
# From our new baserunner_count variable, we were also able to create three more new variables: baserunner_on_first, baserunner_on_second, and baserunner_on_third. Baserunner_on_first determines if there was a baserunner on first base prior to the pitch (indicated by a value of 1). Baserunner_on_second determines if there was a baserunner on second base prior to the pitch (indicated by a value of 1). Baserunner_on_third determines if there was a baserunner on third base prior to the pitch (indicated by a value of 1).
# 
# From our pitch_result variable, we were able to create two more new variables: result_ball and result_strike. Result_ball determines if the result of the pitch was called a ball (indicated by a value of 1). Result_strike determines if the result of the pitch was called a strike (indicated by a value of 1). These two variables will come in handy for some of the visualization types that we will be using in our evaluation.
# 
# From our pitch_type variable, we were able to create six more new variables: pitch_type_CB, pitch_type_CH, pitch_type_CT, pitch_type_FF, pitch_type_FT, and pitch_type_SL. Pitch_type_CB determines if the pitch thrown was a Curveball (indicated by a value of 1). Pitch_type_CH determines if the pitch thrown was a Changeup (indicated by a value of 1). Pitch_type_CT determines if the pitch thrown was a Cutter (indicated by a value of 1). Pitch_type_FF determines if the pitch thrown was a Four-Seam Fastball (indicated by a value of 1). Pitch_type_FT determines if the pitch thrown was a Two-Seam Fastball (indicated by a value of 1). Pitch_type_SL determines if the pitch thrown was a Slider (indicated by a value of 1).
# 
# Now that we have finished cleaning, transforming, and formatting our data, we have identified that we no longer needed the bs_count, game_state, and col_to_be_removed columns, and therefore removed them from our dataset.

# In[6]:


#overview of all of our variables in the pitcher dataframe
# pitcher_df.info()


# After examining our updated dataframe, we now have 33 different variables to work with for further evaluation. Some of these variables may not be needed in our exploratory data analysis.
# 
# Now, we separate our data for each of our pitchers, so that they each have their own dataset that we can focus on in further evaluation.

# In[7]:


#create dataframe of pitcher 1's data only
pitcher1 = pitcher_df[pitcher_df.pitcherid == 1]
#create .csv file of pitcher1 dataframe and save it to the data folder
# pitcher1.to_csv('../data/pitcher1.csv')

#create dataframe of pitcher 2's data only
pitcher2 = pitcher_df[pitcher_df.pitcherid == 2]
#create .csv file of pitcher2 dataframe and save it to the data folder
# pitcher2.to_csv('../data/pitcher2.csv')

#create dataframe of pitcher 3's data only
pitcher3 = pitcher_df[pitcher_df.pitcherid == 3]
#create .csv file of pitcher3 dataframe and save it to the data folder
# pitcher3.to_csv('../data/pitcher3.csv')


# We have created the following three dataframes: pitcher1, pitcher2, and pitcher3. Pitcher1 examines all pitch data for Pitcher 1 only. Pitcher2 examines all pitch data for Pitcher 2 only. Pitcher3 examines all pitch data for Pitcher 3 only.
# 
# Also, before we proceed, it is important to point out that given that we do not have any other results data besides the pitch_result data (i.e. no hit data, or strikeout or walk data), it will be hard to justify the use of any baseball-situation-related variables in our exploratory data analysis phase, as they will not give us much of an indication on why our pitchers struggled or succeeded due to their pitch use. They would, however, be more useful in the modeling phase and also if we had more game or play-by-play data that will give use the end-result of these pitches thrown. These baseball-situation variables include the following variables:
# 
# - ball_count
# - strike_count
# - out_count
# - baserunner_count
# - baserunner_on_first
# - baserunner_on_second
# - baserunner_on_third
# 
# Next, we need to do a quick missing data check on each of our three pitcher dataframes.

# In[8]:


# check the number of missing (NAN) values in each pitcher dataframe
# print(pitcher1.isna().sum())
# print(pitcher2.isna().sum())
# print(pitcher3.isna().sum())


# Given that we just mentioned that we do not need to worry about baseball-situation-related variables at the exact moment, we can ignore the missing values in the baserunner_count, out_count, baserunner_on_first, baserunner_on_second, and baserunner_on_third columns in all three of our pitcher dataframes for now. These rows do not need to be removed, as the other columns in these rows still contain valuable information that we do not want to discard.
# 
# We also notice that some spin_rate observations are missing in each of our three pitcher dataframes. These are such low numbers that we have no problem also keeping these rows. These rows also do not need to be removed, as the other columns in these rows still contain valuable information that we do not want to discard.
# 
# Lastly, we do notice that there are 30 pitch_type observations missing within our pitcher3 dataframe (and as a result, we are missing 30 observations each of pitch_type_CB, pitch_type_CH, pitch_type_CT, pitch_type_FF, pitch_type_FT, and pitch_type_SL). This is not too worrisome either, as it is such a miniscule percentage of observations compared to the total number of observations in the dataframe. These rows also do not need to be removed, as the other columns in these rows still contain valuable information that we do not want to discard. However, given that we know pitch_type will be one of our most used variables in further evaluation, we do need to be mindful that in some areas, we will be missing these 30 observations.
# 
# With that being said, our data cleaning process is now complete and we can move on to exploratory data analysis for each of the three pitchers we are evaluating. Below are links to each of the three pitchers' exploratory data analysis notebooks:
# 
# - [Pitcher 1: Exploratory Data Analysis](https://github.com/michaelpallante/mlb_pitcher_analysis/blob/master/notebooks/mlb_pitcher_analysis_pitcher1_eda.ipynb): 
# <br> This notebook thoroughly examines the data gathered for Pitcher 1 and provides analysis of Pitcher 1's pitch data for both before and after the all-star break.
# - [Pitcher 2: Exploratory Data Analysis](https://github.com/michaelpallante/mlb_pitcher_analysis/blob/master/notebooks/mlb_pitcher_analysis_pitcher2_eda.ipynb): 
# <br> This notebook thoroughly examines the data gathered for Pitcher 2 and provides analysis of Pitcher 2's pitch data for both before and after the all-star break.
# - [Pitcher 3: Exploratory Data Analysis](https://github.com/michaelpallante/mlb_pitcher_analysis/blob/master/notebooks/mlb_pitcher_analysis_pitcher3_eda.ipynb): 
# <br> This notebook thoroughly examines the data gathered for Pitcher 3 and provides analysis of Pitcher 3's pitch data for both before and after the all-star break.
