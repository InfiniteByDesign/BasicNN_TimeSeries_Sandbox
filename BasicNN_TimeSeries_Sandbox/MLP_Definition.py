import numpy as np
import tensorflow as tf

#%% Create a multilayer perceptron network
def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

#%% Train MLP network
def multilayer_perceptron_train(sess,cfg,summary_writer,x,y,keep_prob,x_batches,y_batches,optimizer,cost,merged_summary_op):    
    #  Loop through each epoch
    for epoch in range(cfg.epochs):
        avg_cost = 0.0
        for i in range(x_batches.shape[0]):
            batch_x, batch_y = x_batches[i,:,:], y_batches[i,:,:]
            _, c, summary = sess.run([optimizer, cost, merged_summary_op], 
                            feed_dict={
                                x: batch_x, 
                                y: batch_y, 
                                keep_prob: cfg.dropout_output_keep_prob
                            })
            avg_cost += c / x_batches.shape[0]
        if epoch % cfg.display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()

    
#%% Test teh MLP network
def multiplayer_perceptron_test(sess,cfg,summary_writer,x,keep_prob,x_batches,y_batches,predictions):
    predicted = np.zeros(shape=(y_batches.shape[0],y_batches.shape[1],y_batches.shape[2]), dtype=float)
    for i in range(x_batches.shape[0]):
        batch_testx = x_batches[i,:,:]
        temp = sess.run([predictions], 
                        feed_dict={
                            x: batch_testx,
                            keep_prob: 1.0
                        })
        predicted[i,:,:] = temp[0]
    return np.reshape(predicted,(predicted.shape[0]*predicted.shape[1],y_batches.shape[2]))