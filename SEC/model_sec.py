import tensorflow as tf
import os
from xbrl import XBRLParser, GAAP, GAAPSerializer
num_steps = 500
batch_size = 128

data_x = []
data_y = []

test_x = []
test_y = []
for subdir, dirs, files in os.walk("data/"):
	print(os.path.split(subdir)[-1])
	for f in files:
		xbrl_parser = XBRLParser()
		print(subdir+"/"+f)
		try:
			xbrl = xbrl_parser.parse(subdir+"/"+f)
		except:
			continue
		gaap_obj = xbrl_parser.parseGAAP(xbrl, context="current", ignore_errors=0)
		print("We here 2")
		serializer = GAAPSerializer()
		result = serializer.dump(gaap_obj)
                if "-13-" in f :
		    test_x.append([])
                    for i, k in result.data:
                            test_x[-1].append(k)
                else:
		    data_x.append([])
                    for i, k in result.data:
                            data_x[-1].append(k)
		print(data_x)
		




X = tf.placeholder("float", [None, 76])
Y = tf.placeholder("float", [None, 2])


in_dim = 68

weights = {
    'h1': tf.Variable(tf.random_normal([in_dim, 75])),
    'h2': tf.Variable(tf.random_normal([75, 75])),
    'h3': tf.Variable(tf.random_normal([75, 75])),
    'h4': tf.Variable(tf.random_normal([75, 75])),
    'h5': tf.Variable(tf.random_normal([75, 75])),
    'h6': tf.Variable(tf.random_normal([75, 75])),
    'h7': tf.Variable(tf.random_normal([75, 75])),
    'h8': tf.Variable(tf.random_normal([75, 75])),
    'out': tf.Variable(tf.random_normal([75, 2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([75])),
    'b2': tf.Variable(tf.random_normal([75])),
    'b3': tf.Variable(tf.random_normal([75])),
    'b4': tf.Variable(tf.random_normal([75])),
    'b5': tf.Variable(tf.random_normal([75])),
    'b6': tf.Variable(tf.random_normal([75])),
    'b7': tf.Variable(tf.random_normal([75])),
    'b8': tf.Variable(tf.random_normal([75])),
    'out': tf.Variable(tf.random_normal([2]))
}


l1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
l2 = tf.add(tf.matmul(l1, weights['h1']), biases['b1'])
l3 = tf.add(tf.matmul(l2, weights['h1']), biases['b1'])
l4 = tf.add(tf.matmul(l3, weights['h1']), biases['b1'])
l5 = tf.add(tf.matmul(l4, weights['h1']), biases['b1'])
l6 = tf.add(tf.matmul(l5, weights['h1']), biases['b1'])
l7 = tf.add(tf.matmul(l6, weights['h1']), biases['b1'])
l_out = tf.matmul(l7, weights['out']) + biases['out']

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(data_x)/batch_size)
        x = data_x
        y = data_y
        for i in range(total_batch):
            batch_x, batch_y = data_x[:i], data_y[:i]
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            avg_cost += c / total_batch
        if epoch % 100 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

    pred = tf.nn.softmax(logits)  
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    print(pred)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({X: data_x, Y: data_y}))

