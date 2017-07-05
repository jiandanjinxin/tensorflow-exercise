#http://blog.csdn.net/jiandanjinxin/article/details/74188348
#https://www.tensorflow.org/get_started/mnist/beginners
"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data') # 把数据放在/tmp/data文件夹中

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)   # 读取数据集
#标签数据是"one-hot vectors"。 一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。

# 建立抽象模型
x = tf.placeholder(tf.float32, [None, 784]) # 占位符,
#进行模型计算，a是预测，y 是实际
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
a = tf.nn.softmax(tf.matmul(x, W) + b)

-------------------------
#输入图像２８X28(这个数组展开成一个向量,长度是 28x28 = 784。如何展开这个数组(数字间的顺序)不重要,只要保持各个图片采用相同的方式展开,展平图片的数字数组会丢失图片的二维结构信息。)
#标签数据是"one-hot vectors"。 一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。
------------------------------------


# 定义损失函数和训练方法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))  # 损失函数为交叉熵
optimizer = tf.train.GradientDescentOptimizer(0.5) # 梯度下降法，学习速率为0.5
train = optimizer.minimize(cross_entropy) # 训练目标：最小化损失函数

#成本函数是“交叉熵”(cross-entropy)。交叉熵产生于信息论里面的信息压缩编码技术,但是它后来演变成为从博弈论到机器学习等其他领域里的重要技术手段.比较粗糙的理解是,交叉熵是用来衡量我们的预测用于描述真相的低效性.
--------------------------
# Test trained model
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#tf.argmax 是一个非常有用的函数,它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成,因此最大值1所在的索引位置就是类别标签,比如 tf.argmax(y,1) 返回的是模型对于任一输入x预测到的标签值
#tf.equal 来检测我们的预测是否真实标签匹配
---------------------
# Train
sess = tf.InteractiveSession()      # 建立交互式会话
tf.initialize_all_variables().run()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train.run({x: batch_xs, y: batch_ys})
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))