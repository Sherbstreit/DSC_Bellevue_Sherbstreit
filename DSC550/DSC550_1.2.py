#!/usr/bin/env python
# coding: utf-8

# In[173]:


# Import packages
import numpy as np
import pandas as pd


# In[174]:


# Create array
array = np.random.randint(1,501, size=(10,50))


# In[175]:


print(array)


# In[176]:


# Change array to df
df = pd.DataFrame(data=array)


# In[177]:


print(df)


# In[178]:


# sum by rows and columns
col_sum = df.sum(axis=0)
row_sum = df.sum(axis=1)


# In[179]:


print("The sum of each row is \n", row_sum)


# In[180]:


print("The sum of each column is \n", col_sum)


# In[181]:


# sum of all values
total_sum = df.values.sum()


# In[182]:


print("The total sum of all values in the dataframe is \n", total_sum)


# In[183]:


# find minimum value
min_val = df.values.min()


# In[184]:


print("The minimum value in the dataframe is \n", min_val)


# In[185]:


# find maximum value
max_val = df.values.max()


# In[186]:


print("The maximum value in the dataframe is \n", max_val)


# In[187]:


# find average value
mean_val = df.values.mean()


# In[188]:


print("The average of all values in the dataframe is \n", mean_val)


# In[189]:


# find standard deviation
std_val = df.values.std()


# In[190]:


print("The standard deviation of all values in the dataframe is \n", std_val)


# In[191]:


# Sort by row 2
row2_sort = df.sort_values(by=1, axis=1)


# In[192]:


print("Dataframe sorted by ascending values in row 2 \n", row2_sort)


# In[193]:


# Sort by column 2
col2_sort = df.sort_values(by=1, axis=0)


# In[194]:


print("Dataframe sorted by ascending values in column 2 \n", col2_sort)


# In[ ]:




