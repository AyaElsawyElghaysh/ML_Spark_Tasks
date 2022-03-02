#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
#read csv file 
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df_score=pd.read_csv(url)
df_score


# # datapresprocessing   i will use the data without normalization then we compare the result after we normalize the data 

# # realation between hours and scores
# 
# we will use scatter plot to show the correlation between hours and scores

# In[2]:


import matplotlib.pyplot as plt
plt.scatter(df_score["Hours"],df_score["Scores"])  
plt.title("Scores with hours")
plt.xlabel("hours")
plt.ylabel("Scores")
plt.show()


# # from the previous polt scores is increasing while number of hours for studing is creasing "positive correlation"

# ### Now we separate input data rather than label data
# 
# 

# In[3]:


hours= df_score.iloc[:, :-1].values  
Score= df_score.iloc[:, 1].values 


# In[4]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(hours,Score, 
                            test_size=0.1, random_state=0)      #10% for test to 90% for training


# # Now let`s go to train Linear regrassion model

# In[5]:


lg=LinearRegression()
train_model=lg.fit(X_train,y_train)
print("completed..")


# In[6]:


#we will fil line regression 
line = lg.coef_*hours+lg.intercept_     #mx+b

# Plotting for the test data
plt.scatter(hours, Score)
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.plot(hours, line);
plt.show()


# # try to predict the training text data    x_test

# In[7]:


print(X_test) # Testing data 
y_pred = lg.predict(X_test) # Predicting the scores
print("-----------------")
print(y_pred)


# # try to see the predicted scores in dataframe

# In[8]:


df={"score":y_test,"predicted":y_pred}
data_score=pd.DataFrame(df)
data_score


# # According to our analysis we will predict the presentage of score when hours for studing is 9.25

# In[9]:


hour=[9.25]
score_pred = lg.predict([hour])
print(score_pred)     #make it sense 


# # Then we can evaluate the model  we can use mean abslute error  or mean square error

# In[10]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# # Becuase of the diversity range in score and hours we will use Normalization

# # Normaization 
#  

# In[11]:


max_hour=df_score["Hours"].max()    #9.2
min_hour=df_score["Hours"].min()    #1.1
max_score=df_score["Scores"].max()    
min_score=df_score["Scores"].min()


# In[12]:


df_score["Hours"]=abs(df_score["Hours"]-max_hour)/abs(max_hour-min_hour)
df_score["Scores"]=abs(df_score["Scores"]-max_score)/abs(max_score-min_score)


# In[13]:


hours= df_score.iloc[:, :-1].values  
Score= df_score.iloc[:, 1].values 


# In[14]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(hours,Score, 
                            test_size=0.1, random_state=0)  


# In[15]:


lg=LinearRegression()
train_model=lg.fit(X_train,y_train)
print("completed..")


# In[16]:


#we will fil line regression 
line = lg.coef_*hours+lg.intercept_     #mx+b

# Plotting for the test data
plt.scatter(hours, Score)
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.plot(hours, line);
plt.show()


# In[17]:


print(X_test) # Testing data 
y_pred = lg.predict(X_test) # Predicting the scores
print("-----------------")
print(y_pred)


# In[18]:


df={"score":y_test,"predicted":y_pred}
data_score=pd.DataFrame(df)
data_score


# In[19]:


hour=[9.25]
score_pred = lg.predict([hour])
print(score_pred)     #make it sense 


# In[20]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




