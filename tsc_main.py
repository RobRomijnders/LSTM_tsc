"""
LSTM for time series classification
Made: 30 march 2016

This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
from tsc_model import Model,sample_batch,load_data,check_test

"""Hyperparamaters"""

config = {}                             #Put all configuration information into the dict
config['num_layers'] = 3                #number of layers of stacked RNN's
config['hidden_size'] = 120              #memory cells in a layer
config['max_grad_norm'] = 5             #maximum gradient norm during training
config['batch_size'] = batch_size = 30  
config['learning_rate'] = .005

max_iterations = 3000
dropout = 0.8
ratio = np.array([0.8,0.9]) #Ratios where to split the training and validation set

"""Load the data"""
direc = '/home/rob/Dropbox/ml_projects/LSTM/UCR_TS_Archive_2015'
X_train,X_val,X_test,y_train,y_val,y_test = load_data(direc,ratio,dataset='ChlorineConcentration')
N,sl = X_train.shape
config['sl'] = sl = X_train.shape[1]
config['num_classes'] = num_classes = len(np.unique(y_train))

# Collect the costs in a numpy fashion
epochs = np.floor(batch_size*max_iterations / N)
print('Train %.0f samples in approximately %d epochs' %(N,epochs))
perf_collect = np.zeros((4,int(np.floor(max_iterations /100))))

#Instantiate a model
model = Model(config)



"""Session time"""
sess = tf.Session() #Depending on your use, do not forget to close the session
writer = tf.summary.FileWriter("/home/rob/Dropbox/ml_projects/LSTM/log_tb", sess.graph)  #writer for Tensorboard
sess.run(model.init_op)

step = 0
cost_train_ma = -np.log(1/float(num_classes)+1e-9)  #Moving average training cost
acc_train_ma = 0.0
try:
  for i in range(max_iterations):
    X_batch, y_batch = sample_batch(X_train,y_train,batch_size)

    #Next line does the actual training
    cost_train, acc_train,_ = sess.run([model.cost,model.accuracy, model.train_op],feed_dict = {model.input: X_batch,model.labels: y_batch,model.keep_prob:dropout})
    cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
    acc_train_ma = acc_train_ma*0.99 + acc_train*0.01
    if i%100 == 0:
      #Evaluate training performance
      perf_collect[0,step] = cost_train
      perf_collect[1,step] = acc_train

      #Evaluate validation performance
      X_batch, y_batch = sample_batch(X_val,y_val,batch_size)
      cost_val, summ,acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
      perf_collect[1,step] = cost_val
      perf_collect[2,step] = acc_val
      print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' %(i,max_iterations,cost_train,cost_val,cost_train_ma,acc_train,acc_val,acc_train_ma))

      #Write information to TensorBoard
      writer.add_summary(summ, i)
      writer.flush()

      step +=1
except KeyboardInterrupt:
  #Pressing ctrl-c will end training. This try-except ensures we still plot the performance
  pass
  
acc_test,cost_test = check_test(model,sess,X_test,y_test)
epoch = float(i)*batch_size/N
print('After training %.1f epochs, test accuracy is %5.3f and test cost is %5.3f'%(epoch,acc_test,cost_test))

"""Additional plots"""
plt.plot(perf_collect[0],label='Train')
plt.plot(perf_collect[1],label = 'Valid')
plt.plot(perf_collect[2],label = 'Valid accuracy')
plt.axis([0, step, 0, np.max(perf_collect)])
plt.legend()
plt.show()



