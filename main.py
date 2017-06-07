# coding = utf-8


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

NUM_TRAIN = 20000
BATCH_SIZE = 100

# MNISTデータセットを読み込む
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# プレイスホルダ作成
x = tf.placeholder(tf.float32,[None, 784])

# 重み
W = tf.Variable(tf.zeros([784, 10]))

# バイアス
b = tf.Variable(tf.zeros([10]))

#　ソフトマックス
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 出力(予測値)
y_ = tf.placeholder(tf.float32, [None, 10])


# 交差エントロピーの最小化
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# 精度確認
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 初期化
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(NUM_TRAIN):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 1000 == 0:
        a = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
        print("Accuracy: " + str(a) + " %")
