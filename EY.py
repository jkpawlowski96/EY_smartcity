#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime


# In[2]:


d={
    'hash'          :'category',
    'trajectory_id' :'category',
    'time_entry'    :'category',
    'time_exit'    :'category',
    'vmax':'float64',
    'vmin':'float64',
    'vmean':'float64',
    'x_entry':'float64',
    'y_entry':'float64',
    'x_exit':'float64',
    'y_exit':'float64'
}

train = pd.read_csv('./input/data_train.csv',nrows=10000,dtype=d).drop(columns=['Unnamed: 0'])
test = pd.read_csv('./input/data_test.csv',nrows=10000,dtype=d).drop(columns=['Unnamed: 0'])


# In[3]:


def direct(ax, ay, bx, by):
    return bx-ax, by-ay


def calc_directory(v):
    # 7,8 ->9,10 
    return direct(v[7],v[8],v[9],v[10])


def add_directories(dirx, diry, indexes, df):
    first = True
    meanx, meany = 0, 0
    for i in indexes:
        if first==True:
            first = False
            dirx[i], diry[i] = calc_directory(df.values[i])
            
        else:    
            x, y = calc_directory(df.values[last])
            meanx, meany = (x+meanx)/2, (y+meany)/2
            dirx[i], diry[i] = meanx, meany
            
        last = i
        # print(dirx[i], diry[i])
      
        
def make_directories(df):
    dirx=np.empty(len(df))
    diry=np.empty(len(df))
    drivers = df.hash.unique()
    """
    for driver in drivers[:100]:
        indexes= train[train.hash == driver].index
        add_directories(dir, indexes, df)
    """
    for driver in drivers:
        add_directories(dirx, diry, df[df.hash == driver].index, df) 
    
    df['x_dir'] = dirx
    df['y_dir'] = diry

    
make_directories(train)
make_directories(test)


# In[4]:


def find_last_trajectories(df):
    drivers = df.hash.unique()
    df2 = pd.DataFrame(columns=df.columns)
    for driver in drivers:
        df2 = df2.append(df[df.hash == driver][-1:])
    df2 = df2.reset_index(drop=True)    
    df2.to_csv('./input/test2.csv',index=False)
       

find_last_trajectories(test)

test2 = pd.read_csv('./input/test2.csv')
trajectory_id = test2['trajectory_id']


# In[5]:


train.info()


# In[6]:


train[:10]


# In[7]:


test[:10]


# In[8]:


test2[:10]


# In[9]:


def within_measure(x, y):
    #  3750901.5068 ≤ 𝑥 ≤ 3770901.5068
    #  −19268905.6133 ≤ 𝑦 ≤ −19208905.6133
    if 3750901.5068 <= x and x <= 3770901.5068 and -19268905.6133 <= y and y <= -19208905.6133:
        return 1
    else:
        return 0

def time_to_sec(values):
    return pd.to_timedelta(values).dt.total_seconds().astype(int)

train.time_entry = time_to_sec(train.time_entry)
train.time_exit = time_to_sec(train.time_exit)
test.time_entry = time_to_sec(test.time_entry)
test.time_exit = time_to_sec(test.time_exit)
test2.time_entry = time_to_sec(test.time_entry)
test2.time_exit = time_to_sec(test.time_exit)


# In[10]:


"""
X = train['x_exit'].values 
Y = train['y_exit'].values
train.drop(columns=['x_exit','y_exit'],inplace=True)
test = test2
test.drop(columns=['x_exit','y_exit'],inplace=True)

city = [within_measure(x, y) for x,y in zip(X,Y)]
"""
y_train=pd.DataFrame() 
y_train['x_exit']= train['x_exit']
y_train['y_exit']= train['y_exit']

train.drop(columns=['x_exit','y_exit'],inplace=True)
test2.drop(columns=['x_exit','y_exit'],inplace=True)


# In[11]:



for col in ['time_entry','time_exit','vmax','vmin','vmean','x_entry','y_entry','x_dir','y_dir']:
    mode = train[col].mode()[0]
    test[col].fillna(mode, inplace=True)
    test2[col].fillna(mode, inplace=True)
    train[col].fillna(mode, inplace=True)


# In[12]:


from sklearn.neural_network import MLPRegressor


x_train = train.drop(columns=['hash','trajectory_id'])
x_test = test2.drop(columns=['hash','trajectory_id'])

model = MLPRegressor(hidden_layer_sizes=[100],
                      learning_rate_init=.01,
                      max_iter=200,
                      learning_rate='constant',
                      solver='adam',
                      activation='relu',
                     verbose=True)
model.fit(x_train, y_train)


# In[13]:


pred = model.predict(x_test)
#pred = pd.DataFrame(data=pred,columns=['x_exit','y_exit'])
pred = pd.DataFrame({'x_exit':pred[:,0],'y_exit':pred[:,1]})


# In[14]:


X = pred['x_exit'].values 
Y = pred['y_exit'].values

city = [within_measure(x, y) for x,y in zip(X,Y)]


# In[15]:


submission= pd.DataFrame()
submission['id']=trajectory_id 
submission['target'] = city
submission.to_csv('./output/submission.csv',index=False)    
submission[:20]


# In[16]:


min(city)


# In[17]:


max(city)

