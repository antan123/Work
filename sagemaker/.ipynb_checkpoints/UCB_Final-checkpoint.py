
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


raw_data = pd.ExcelFile('rbull1.xlsx')


# In[3]:


raw_data.sheet_names


# In[4]:


data = raw_data.parse('rbull1.csv')


# In[5]:


data.head()


# In[6]:


data['ConversionperCost'] = (data['conversions']/data['cost']).replace([np.inf, -np.inf], np.nan)


# In[7]:


data.head()


# In[8]:


opt_data = (data[['date','id','ConversionperCost']].copy(deep = True))


# In[9]:


opt_data.reset_index(inplace = True)


# In[10]:


opt_data.pivot(columns='id',values='ConversionperCost').to_csv('output.csv')


# In[11]:


import gc
del opt_data
del raw_data
del data
gc.collect()


# In[12]:


data = pd.read_csv('output.csv')


# In[13]:


data.fillna(0, inplace = True)


# In[14]:


data.head()


# In[ ]:


import math
N = data.shape[0]
d = data.shape[1]
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    print("Experiment nnumber {}".format(n))
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = data.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

