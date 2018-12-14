from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
def preprocess(name, X, Y):
    f = open("./Twitter_sentiment_DJIA30/twitter_data_"+name+".csv").readlines()
    out = []
    del(f[0])
    for line in f:
        temp = line.replace('\n','').split(',')
        if temp[1] != '' and temp[3] != '':
            temp[1] = temp[1].split('.')[0]
            temp[3] = temp[3].split('.')[0]
            out.append([temp[0],int(temp[3]) - int(temp[1])])
    ff = open("./Twitter_sentiment_DJIA30/financial_data_"+name+".csv").readlines()
    dic = {}
    del(ff[0])
    for line in ff:
        temp = line.replace('\n','').split(',')
        if temp[2] != '' and temp[3] != '':
            temp[2] = temp[2].split('.')[0]
            temp[3] = temp[3].split('.')[0]
            if int(temp[2]) > int(temp[3]):
                flag = np.array([0,1])
            else:
                flag = np.array([1,0])
            dic[temp[0]] = flag
    i = 0
    while(i + 100 < len(out)):
        if out[i + 100][0] in dic.keys():
            temp = []
            for j in range(i,i+100):
                temp.append(np.array(out[j][1]))
            temp = np.array(temp)
            X.append(temp.reshape(10,10))
            Y.append(dic[out[i+100][0]])
        i += 1
    return X,Y

names = ['AXP','PFE','IBM','BA','PG','INTC','CAT','T','JNJ','CSCO','TRV','JPM','CVX','UNH','KO','DD','UTX','MCD','DIS','V','MMM','VZ','MRK','GE','WMT','MSFT','GS','XOM','NKE','HD']
X = []
Y = []
for name in names:
    X,Y = preprocess(name, X, Y)
x_train = np.array(X)
y_train = np.array(Y)


learning_rate = 0.001
training_steps = 1000
batch_size = 128
display_step = 10
num_input = 10
timesteps = 10
num_hidden = 128 
num_classes = 2

X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden,num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}
def RNN(x, weights, biases):
    x = tf.unstack(x, timesteps , 1)

    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']
logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    num_of_batches = int(int(len(x_train)) / batch_size)
    for step in range(1, training_steps+1):
        ptr = 0
        for j in range(num_of_batches):
            batch_x, batch_y = x_train[ptr:ptr+batch_size], y_train[ptr:ptr+batch_size]
            ptr += batch_size
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
