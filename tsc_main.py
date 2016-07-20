

"""LSTM for time series classification
Made: 30 march 2016

This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops


def sample_batch(X_train,y_train,batch_size,num_steps):
    """ Function to sample a batch for training"""
    N,data_len = X_train.shape
    ind_N = np.random.choice(N,batch_size,replace=False)
    ind_start = np.random.choice(data_len-num_steps,1)
    #form batch
    X_batch = X_train[ind_N,ind_start:ind_start+num_steps]
    y_batch = y_train[ind_N]
    return X_batch,y_batch

def check_test(X_test,y_test,batch_size,num_steps):
    """ Function to check the test_accuracy on the entire test set
    This is a workaround. I haven't figured out yet how to make the graph
    general for multiple batch sizes."""
    N = X_test.shape[0]
    num_batch = np.floor(N/batch_size)
    test_acc = np.zeros(num_batch)
    for i in range(int(num_batch)):
      X_batch, y_batch = sample_batch(X_test,y_test,batch_size,num_steps)
      test_acc[i] = session.run(accuracy,feed_dict = {input_data: X_batch, targets: y_batch, initial_state:state,keep_prob:1})
    return np.mean(test_acc)




"""Hyperparamaters"""
init_scale = 0.08           #Initial scale for the states
max_grad_norm = 25          #Clipping of the gradient before update
num_layers = 2              #Number of stacked LSTM layers
num_steps = 64              #Number of steps to backprop over at every batch
hidden_size = 13            #Number of entries of the cell state of the LSTM
max_iterations = 2000       #Maximum iterations to train
batch_size = 30             #Batch size
dropout = 0.8               # Keep probability of the dropout wrapper


"""Place holders"""
input_data = tf.placeholder(tf.float32, [None, num_steps], name = 'input_data')
targets = tf.placeholder(tf.int64, [None], name='Targets')
#Used later on for drop_out. At testtime, we pass 1.0
keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob')

initializer = tf.random_uniform_initializer(-init_scale,init_scale)
with tf.variable_scope("model", initializer=initializer):
  """Define the basis LSTM"""
  with tf.name_scope("LSTM_setup") as scope:
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)   #Initialize the zero_state. Note that it has to be run in session-time
    #We have only one input dimension, but we generalize our code for future expansion
    inputs = tf.expand_dims(input_data, 2)

  #Define the recurrent nature of the LSTM
  #Re-use variables only after first time-step
  with tf.name_scope("LSTM") as scope:
    outputs = []
    state = initial_state
    with tf.variable_scope("LSTM_state"):
      for time_step in range(num_steps):
       if time_step > 0: tf.get_variable_scope().reuse_variables()
       (cell_output, state) = cell(inputs[:, time_step, :], state)
       outputs.append(cell_output)       #Now cell_output is size [batch_size x hidden_size]
    avg_output = tf.reduce_mean(tf.pack(outputs),0)
    size1 = tf.shape(avg_output)
    size2 = tf.shape(cell_output)


#Generate a classification from the last cell_output
#Note, this is where timeseries classification differs from sequence to sequence
#modelling. We only output to Softmax at last time step
with tf.name_scope("Softmax") as scope:
  with tf.variable_scope("Softmax_params"):
    # Both datasets have four output classes. Improve the code by changing the 4
    # into a hyperparameter
    softmax_w = tf.get_variable("softmax_w", [hidden_size, 4])
    softmax_b = tf.get_variable("softmax_b", [4])
  logits = tf.matmul(avg_output, softmax_w) + softmax_b
  #Use sparse Softmax because we have mutually exclusive classes
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,targets,name = 'Sparse_softmax')
  cost = tf.reduce_sum(loss) / batch_size
  #Pass on a summary to Tensorboard
  cost_summ = tf.scalar_summary('Cost',cost)
  # Calculate the accuracy
  correct_prediction = tf.equal(tf.argmax(logits,1), targets)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  accuracy_summary = tf.scalar_summary("accuracy", accuracy)


"""Optimizer"""
with tf.name_scope("Optimizer") as scope:
  tvars = tf.trainable_variables()
  #We clip the gradients to prevent explosion
  grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),max_grad_norm)
  optimizer = tf.train.AdamOptimizer(8e-3)
  gradients = zip(grads, tvars)
  train_op = optimizer.apply_gradients(gradients)
  # Add histograms for variables, gradients and gradient norms.
  # The for-loop loops over all entries of the gradient and plots
  # a histogram. We cut of
  for gradient, variable in gradients:
    if isinstance(gradient, ops.IndexedSlices):
      grad_values = gradient.values
    else:
      grad_values = gradient
    h1 = tf.histogram_summary(variable.name, variable)
    h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
    h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

"""Load the data"""
dummy = True
if dummy:
  data_train = np.loadtxt('UCR_TS_Archive_2015/Two_Patterns/Two_Patterns_TRAIN',delimiter=',')
  data_test_val = np.loadtxt('UCR_TS_Archive_2015/Two_Patterns/Two_Patterns_TEST',delimiter=',')
else:
  data_train = np.loadtxt('data_train_dummy',delimiter=',')
  data_test_val = np.loadtxt('data_test_dummy',delimiter=',')
data_test,data_val = np.split(data_test_val,2)
X_train = data_train[:,1:]
X_val = data_val[:,1:]
X_test = data_test[:,1:]
N = X_train.shape[0]
Ntest = X_test.shape[0]
# Targets have labels 1-indexed. We subtract one for 0-indexed
y_train = data_train[:,0]-1
y_val = data_val[:,0]-1
y_test = data_test[:,0]-1

#Final code for the TensorBoard
merged = tf.merge_all_summaries()


# Collect the costs in a numpy fashion
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))
perf_collect = np.zeros((3,int(np.floor(max_iterations /100))))

"""Session time"""
with tf.Session() as session:
  writer = tf.train.SummaryWriter("/home/rob/Dropbox/ml_projects/LSTM/log_tb", session.graph_def)
  tf.initialize_all_variables().run()


  step = 0
  for i in range(max_iterations):

    # Calculate some sizes
    N = X_train.shape[0]

    #Sample batch for training
    X_batch, y_batch = sample_batch(X_train,y_train,batch_size,num_steps)
    state = initial_state.eval()  #Fire up the LSTM

    #Next line does the actual training
    session.run(train_op,feed_dict = {input_data: X_batch,targets: y_batch,initial_state: state,keep_prob:dropout})
    if i==0:
        # Uset this line to check before-and-after test accuracy
        acc_test_before = check_test(X_test,y_test,batch_size,num_steps)
        result = session.run([size1,size2],feed_dict = {input_data: X_batch,targets: y_batch,initial_state: state,keep_prob:dropout})
        print(result[0])
        print(result[1])
    if i%100 == 0:
      #Evaluate training performance
      X_batch, y_batch = sample_batch(X_train,y_train,batch_size,num_steps)
      cost_out = session.run(cost,feed_dict = {input_data: X_batch, targets: y_batch, initial_state:state,keep_prob:1})
      perf_collect[0,step] = cost_out
      #print('At %d out of %d train cost is %.3f' %(i,max_iterations,cost_out)) #Uncomment line to follow train cost

      #Evaluate validation performance
      X_batch, y_batch = sample_batch(X_val,y_val,batch_size,num_steps)
      result = session.run([cost,merged,accuracy],feed_dict = {input_data: X_batch, targets: y_batch, initial_state:state,keep_prob:1})
      cost_out = result[0]
      perf_collect[1,step] = cost_out
      acc_val = result[2]
      perf_collect[2,step] = acc_val
      print('At %d out of %d val cost is %.3f and val acc is %.3f' %(i,max_iterations,cost_out,acc_val))

      #Write information to TensorBoard
      summary_str = result[1]
      writer.add_summary(summary_str, i)
      writer.flush()

      step +=1
  acc_test = check_test(X_test,y_test,batch_size,num_steps)

"""Additional plots"""
print('The accuracy on the test data is %.3f, before training was %.3f' %(acc_test,acc_test_before))
plt.plot(perf_collect[0],label='Train')
plt.plot(perf_collect[1],label = 'Valid')
plt.plot(perf_collect[2],label = 'Valid accuracy')
plt.axis([0, step, 0, np.max(perf_collect)])
plt.legend()
plt.show()



