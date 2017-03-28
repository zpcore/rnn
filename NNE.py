import numpy as np
import pickle
import tensorflow as tf
from data_construct import data_construct
import data_gen

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 2#100
display_step = 1

# Network Parameters
n_hidden_1 = 4 # 1st layer number of features
n_hidden_2 = 4 # 2nd layer number of features
n_input = 8 # MNIST data input (img shape: 28*28)
n_output=1

#test data parameters
predict_horizon = 4

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

# Create model
def multilayer_perceptron(x, weights, biases):
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	# Output layer with linear activation
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer


# Store layers weight & bias
weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost = tf.reduce_mean(tf.nn.l2_loss(t=pred-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Initializing the variables
init = tf.global_variables_initializer()

with open(r'./traindata.log', 'rb') as _load_file:
	train_data = pickle.load(_load_file)

with open(r'./testdata.log', 'rb') as _load_file2:
	test_data = pickle.load(_load_file2)

dc=data_construct(predict_horizon,train_data)
tc=data_construct(predict_horizon,test_data)

# Launch the graph
with tf.Session() as sess:
		sess.run(init)
		# Training cycle
		for epoch in range(training_epochs):
				avg_cost = 0.
				total_batch = int((dc.num_example-predict_horizon+1)/batch_size)
				# Loop over all batches
				for i in range(total_batch):
						batch_x=[]
						batch_y=[]
						batch_x, batch_y = dc.next_batch(batch_x,batch_y,batch_size)
						# Run optimization op (backprop) and cost op (to get loss value)
						_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
						avg_cost += c / total_batch
				# Display logs per epoch step
				if epoch % display_step == 0:
					print "Epoch:", '%04d' % (epoch+1), "cost=", \
						"{:.9f}".format(avg_cost)
				dc.clr_count()

		print "Optimization Finished!"
		print "Testing the Neural Network"

        #for _ in range(10)#10 test data



		# Test model
		#correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		
        # Calculate accuracy
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		#print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})



# def main():
# 	#reload object from the file
# 	with open(r'./td', 'rb') as _load_file:
# 		train_data = pickle.load(_load_file)
# 	#print train_data
# 	dc=data_construct(4,train_data)

# if __name__ == "__main__":
#     main()