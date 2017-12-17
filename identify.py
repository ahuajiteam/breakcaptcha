
import os
import tensorflow as tf
import reader
import time
from tensorflow.examples.tutorials.mnist import input_data

N_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 100
N_EPOCHS = 100
HEIGHT = 80
WIDTH = 120
TrainingDataPath = r'Data/debugdata' #modify here
TestingDataPath = r'Data/testdebugdata' #modify here
import utils

class CNN:
    def __init__(self, n_classes, lr, height, width):
        self.n_classes = n_classes
        self.n_length = 4
        self.learning_rate = lr
        self.height = height
        self.width = width
        self.global_step = tf.Variable(0, dtype = tf.int32, trainable = False)
        self.DROPOUT = 0.95
        self.ckptdir = r'checkpoints\fordebug' #modify here

        self.X = tf.placeholder(tf.float32, [None, self.height * self.width])
        self.Y = tf.placeholder(tf.float32, [None, self.n_length * self.n_classes])
        self.dropout = tf.placeholder(tf.float32)
        w_alpha = 0.01
        b_alpha = 0.1

        x = tf.reshape(self.X, shape=[-1, self.width, self.height, 1])
        w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
        b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.dropout)

        w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.dropout)

        w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.dropout)

        # Fully connected layer
        w_d = tf.Variable(w_alpha*tf.random_normal([int(self.width/8)*int(self.height/8)*64, 1024]))
        b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, self.dropout)

        w_out = tf.Variable(w_alpha*tf.random_normal([1024, self.n_length*self.n_classes]))
        b_out = tf.Variable(b_alpha*tf.random_normal([self.n_length*self.n_classes]))
        self.output = tf.add(tf.matmul(dense, w_out), b_out)

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.predict = tf.reshape(self.output, [-1, self.n_length, self.n_classes])
        max_idx_p = tf.argmax(self.predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.n_length, self.n_classes]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def train_model(model, data_path, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, skip_step=SKIP_STEP):
    input = reader.ReadAll(data_path) 
    batch_size = min(batch_size, input.total_numbers)
    if (os.path.exists(model.ckptdir) == False):
        os.makedirs(model.ckptdir)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(model.ckptdir + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            # print ("found!")
            saver.restore(sess, ckpt.model_checkpoint_path)

        initial_step = model.global_step.eval()
        start_time = time.time()
        n_batches = int(input.total_numbers / batch_size)  
        total_loss = 0

        print ('total_numbers = {}'.format(input.total_numbers))
        # print (initial_step)

        for i in range(initial_step, n_batches * n_epochs):
            X_batch, Y_batch = input.getbatch(i, batch_size, model.width, model.height, "ONLY_NUMBERS")
            _, loss_batch = sess.run([model.optimizer, model.loss],
                                     feed_dict={model.X: X_batch, model.Y: Y_batch, model.dropout: model.DROPOUT})
            total_loss += loss_batch
            print('loss at step {}: {:5.6f}'.format(i + 1, loss_batch))

            if (i + 1) % skip_step == 0:
                batch_x_test, batch_y_test = input.getbatch(i + 1, min(100, input.total_numbers), model.width, model.height, "ONLY_NUMBERS")
                acc = sess.run(model.accuracy, feed_dict={model.X: batch_x_test, model.Y: batch_y_test, model.dropout: 1.})
                print('Average loss at step {}: {:5.6f}, Accuracy of all numbers = {}'.format(i + 1, total_loss / skip_step, acc))
                total_loss = 0.0
                saver.save(sess, model.ckptdir + '/identify-convnet', i)
                # if (acc > 0.95):
                #     break 

        print("Optimization Finished!")  # should be around 0.35 after 25 epochs
        print("Total time: {0} seconds".format(time.time() - start_time))

def test_model(model, data_path):
    input = reader.ReadAll(data_path) ##need modify
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(model.ckptdir + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print ("checkpoint not found")
            exit(1)

        batch_x_test, batch_y_test = input.getbatch(0, input.total_numbers, model.width, model.height, "ONLY_NUMBERS")
        preds = sess.run(model.output, feed_dict={model.X: batch_x_test, model.Y: batch_y_test, model.dropout: 1.0})

        preds = tf.argmax(tf.reshape(preds, [-1, model.n_length, model.n_classes]), 2)
        y = tf.argmax(tf.reshape(batch_y_test, [-1, model.n_length, model.n_classes]), 2)
        correct_pred = tf.equal(y, preds).eval()
        # print (correct_pred)
        
        acc = 0
        for i in range(input.total_numbers):
            acc += 1
            for j in range(4):
                if (correct_pred[i][j] != True):
                    acc -= 1
                    break

        print("Accuracy on {} {:5.6f}".format(data_path, acc / input.total_numbers))    

# def test_one(X):
#     model = CNN(N_CLASSES, LEARNING_RATE, HEIGHT, WIDTH)
#     preds = []
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver()
#         ckpt = tf.train.get_checkpoint_state(os.path.dirname(model.ckptdir + '/checkpoint'))
#         if ckpt and ckpt.model_checkpoint_path:
#             saver.restore(sess, ckpt.model_checkpoint_path)
#         else:
#             print ("checkpoint not found")
#             exit(1)

#         preds = sess.run([model.output], feed_dict={model.X: x, model.dropout: 1.0})

#         preds = tf.reshape(preds, [-1, model.n_length, model.n_classes])
#         preds = tf.argmax(preds, 2).eval()

#     return preds[0]

if __name__ == '__main__':
    model = CNN(N_CLASSES, LEARNING_RATE, HEIGHT, WIDTH)
    train_model(model, TrainingDataPath)
    test_model(model, TrainingDataPath)
    test_model(model, TestingDataPath)