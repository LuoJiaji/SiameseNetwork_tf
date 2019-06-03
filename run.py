
""" Siamese implementation using Tensorflow with MNIST example.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import system things
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

#import helpers
import inference
import visualize
num_classes = 10

def create_rand_batch_pairs(x, digit_indices,batch):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
        
    for d in range(int(batch/2)):
        n = np.random.randint(10)
        ind1 = np.random.randint(len(digit_indices[n]))
        ind2 = np.random.randint(len(digit_indices[n]))
        z1, z2 = digit_indices[n][ind1], digit_indices[n][ind2]
        pairs += [[x[z1], x[z2]]]
        
        dn = np.random.randint(10)
        while dn == n:
            dn = np.random.randint(10)
            
        ind3 = np.random.randint(len(digit_indices[dn]))
        z1, z2 = digit_indices[n][ind1], digit_indices[dn][ind3]
        pairs += [[x[z1], x[z2]]]
        labels += [1, 0]    

    return np.array(pairs), np.array(labels)


# prepare data and tf.session
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
test_x = mnist.test.images
test_y = mnist.test.labels
train_x =  mnist.train.images 
train_y = mnist.train.labels


train_digit_indices = [np.where(train_y == i)[0] for i in range(num_classes)]
train_pairs, train_y = create_rand_batch_pairs(train_x, train_digit_indices,256)
#plt.subplot(2,2,1)
#plt.imshow(train_pairs[0,0,:].reshape((28, 28)), cmap='gray')
#plt.subplot(2,2,2)
#plt.imshow(train_pairs[0,1,:].reshape((28, 28)), cmap='gray')
#plt.subplot(2,2,3)
#plt.imshow(train_pairs[1,0,:].reshape((28, 28)), cmap='gray')
#plt.subplot(2,2,4)
#plt.imshow(train_pairs[1,1,:].reshape((28, 28)), cmap='gray')
#plt.show()
test_digit_indices = [np.where(test_y == i)[0] for i in range(num_classes)]


sess = tf.InteractiveSession()

# setup siamese network
siamese = inference.siamese()
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
train_step = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9, epsilon=1e-06).minimize(siamese.loss)
saver = tf.train.Saver()
tf.global_variables_initializer().run()

# if you just want to load a previously trainmodel?
# new = False
new = True

model_ckpt = './model/model.ckpt'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False

# start training

if new:
    for step in range(50000):
#        batch_x1, batch_y1 = mnist.train.next_batch(128)
#        batch_x2, batch_y2 = mnist.train.next_batch(128)
#        batch_y = (batch_y1 == batch_y2).astype('float')
        
        train_pairs, train_y = create_rand_batch_pairs(train_x, train_digit_indices,256)
        batch_x1 = train_pairs[:,0]
        batch_x2 = train_pairs[:,1]
        batch_y = train_y.astype('float')
        
        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                            siamese.x1: batch_x1, 
                            siamese.x2: batch_x2, 
                            siamese.y_: batch_y})
        # pre = sess.run(siamese.pre, feed_dict={siamese.x1: batch_x1, 
        #                     siamese.x2: batch_x2})

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()

        if step % 100 == 0:
            # print ('step %d: loss %.3f' % (step, loss_v))
            print('step:',step, 'loss:',loss_v)

        if (step+1) % 1000 == 0 :
            saver.save(sess, './model/model.ckpt')
            # embed = siamese.o1.eval({siamese.x1: mnist.test.images})
            # embed.tofile('embed.txt')
                
            
            
            result = []                      
            for i in range(len(test_x)):
                test_pairs=[]
#                if i%1000 == 0:
#                    print(i)
                for j in range(num_classes):
        #            a = x_test[digit_indices[0][0]]
                    a = test_x[i]
                    b = test_x[test_digit_indices[j][10]]
                    test_pairs += [[a,b]]
                test_pairs = np.array(test_pairs)
                
                pre = sess.run(siamese.pre, 
                                feed_dict={siamese.x1: test_pairs[:,0], siamese.x2: test_pairs[:,1]})
                
                result += [pre]
            result = np.array(result)
            pre = np.argmin(result,axis=1)
            acc = np.mean(pre==test_y)
            print('test accuracy:',acc)
            
            
else:
    saver.restore(sess, './model/model.ckpt')

# visualize result
# embed = siamese.o1.eval({siamese.x1: mnist.test.images})
# x_test = mnist.test.images.reshape([-1, 28, 28])
# visualize.visualize(embed, x_test)
