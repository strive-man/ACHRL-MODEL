# import tensorflow as tf
# import numpy as np

# #构造输入数据（我们用神经网络拟合x_data和y_data之间的关系）
# x_data = np.linspace(-1,1,300)[:, np.newaxis] #-1到1等分300份形成的二维矩阵
# noise = np.random.normal(0,0.05, x_data.shape) #噪音，形状同x_data在0-0.05符合正态分布的小数
# y_data = np.square(x_data)-0.5+noise #x_data平方，减0.05，再加噪音值
#
# #输入层（1个神经元）
# xs = tf.placeholder(tf.float32, [None, 1]) #占位符，None表示n*1维矩阵，其中n不确定
# ys = tf.placeholder(tf.float32, [None, 1]) #占位符，None表示n*1维矩阵，其中n不确定
#
# #隐层（10个神经元）
# W1 = tf.Variable(tf.random_normal([1,10])) #权重，1*10的矩阵，并用符合正态分布的随机数填充
# b1 = tf.Variable(tf.zeros([1,10])+0.1) #偏置，1*10的矩阵，使用0.1填充
# Wx_plus_b1 = tf.matmul(xs,W1) + b1 #矩阵xs和W1相乘，然后加上偏置
# output1 = tf.nn.relu(Wx_plus_b1) #激活函数使用tf.nn.relu
#
# #输出层（1个神经元）
# W2 = tf.Variable(tf.random_normal([10,1]))
# b2 = tf.Variable(tf.zeros([1,1])+0.1)
# Wx_plus_b2 = tf.matmul(output1,W2) + b2
# output2 = Wx_plus_b2
#
# #损失
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-output2),reduction_indices=[1])) #在第一维上，偏差平方后求和，再求平均值，来计算损失
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 使用梯度下降法，设置步长0.1，来最小化损失
#
# #初始化
# init = tf.global_variables_initializer() #初始化所有变量
# sess = tf.Session()
# sess.run(init) #变量初始化
#
# #训练
# for i in range(1000): #训练1000次
#     _,loss_value = sess.run([train_step,loss],feed_dict={xs:x_data,ys:y_data}) #进行梯度下降运算，并计算每一步的损失
#     if(i%50==0):
#         print(loss_value) # 每50步输出一次损失
# # ············································································································
# import tensorflow as tf
# import numpy as np
# max_steps=1000
# learning_rate=0.001 #学习率
# dropout=0.9 #保留的数据
# log_dir='./logs/xyhuashuguanxi'
#
# sess=tf.InteractiveSession()
#
# #构造输入数据（我们用神经网络拟合x_data和y_data之间的关系）
# x_data = np.linspace(-1,1,300)[:, np.newaxis] #-1到1等分300份形成的二维矩阵
# noise = np.random.normal(0,0.05, x_data.shape) #噪音，形状同x_data在0-0.05符合正态分布的小数
# y_data = np.square(x_data)-0.5+noise #x_data平方，减0.05，再加噪音值
#
# with tf.name_scope('input'):#with块中名字才是最重要的一个块
#     #输入层（1个神经元）
#     x = tf.placeholder(tf.float32, [None, 1]) #占位符，None表示n*1维矩阵，其中n不确定
#     y = tf.placeholder(tf.float32, [None, 1]) #占位符，None表示n*1维矩阵，其中n不确定
#
#
# #定义神经网络的初始化方法
# def weight_varible():
#     # initial=tf.truncated_normal(shape,stddev=0.1) #截断正态分布 这里可以用he_initinelize
#     initial =tf.Variable(tf.random_normal([1, 10]))  # 权重，1*10的矩阵，并用符合正态分布的随机数填充
#     return tf.Variable(initial) #创建一个变量
#
# def bias_variable(): #截距
#     # initial = tf.constant(0.1,shape=shape)
#     initial= tf.Variable(tf.zeros([1, 10]) + 0.1)
#     return tf.Variable(initial)
#
# #以下代码是关于画图的，
# # 定义variable变量的数据汇总函数，我们计算出变量的mean、stddev、max、min
# #对这些标量数据使用tf.summary.scalar进行记录和汇总，使用tf.summary.histogram直接记录变量var的直方图数据
# def variable_summaries(var):
#     with tf.name_scope('summaries'):
#         mean=tf.reduce_mean(var)
#         tf.summary.scalar('mean',mean)
#         with tf.name_scope('stddev'):
#             stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
#
#         tf.summary.scalar('stddev',stddev)
#         tf.summary.scalar('max',tf.reduce_max(var))
#         tf.summary.scalar('min',tf.reduce_min(var))
#         tf.summary.histogram('histogram',var)
#
# # 设计一个MLP多层神经网络来训练数据，在每一层中都对模型数据进行汇总
# def nn_layer(input_tensor, layer_name, act=tf.nn.relu):
#     with tf.name_scope(layer_name):
#         # 定义一个隐藏层input_dim上一层 output_dim本层输出
#         with tf.name_scope('weights'):
#             weights = weight_varible()  # shape传进来是上一层输入，
#             # 本层输出如果是MLP，就是全连接可以知道参数个数
#             variable_summaries(weights)  # 把权重的各个中标（方差+平均值）进行总结
#         with tf.name_scope('biases'):
#             biases = bias_variable()
#             variable_summaries(biases)
#         with tf.name_scope('Wx_plus_b'):
#             preactivate = tf.matmul(input_tensor, weights) + biases  # 带到激活函数之前的公式
#             tf.summary.histogram('pre_activations', preactivate)
#             activations = act(preactivate, name='activations')
#             # 运用激活函数 函数里面传函数 高阶函数
#             return activations
#
#
# #隐层（10个神经元）
# hidden1=nn_layer(x,'layer1') #建立第一层 隐藏层
# # with tf.name_scope('dropout'):
# #     keep_prob=tf.placeholder(tf.float32)
# #     tf.summary.scalar('dropout_keep_probability',keep_prob)
# #     dropped=tf.nn.dropout(hidden1,keep_prob) #应用drop_out函数，保留下来的数据
#
# # W1 = tf.Variable(tf.random_normal([1,10])) #权重，1*10的矩阵，并用符合正态分布的随机数填充
# # b1 = tf.Variable(tf.zeros([1,10])+0.1) #偏置，1*10的矩阵，使用0.1填充
# # Wx_plus_b1 = tf.matmul(x,W1) + b1 #矩阵xs和W1相乘，然后加上偏置
# # output1 = tf.nn.relu(Wx_plus_b1) #激活函数使用tf.nn.relu
#
# #然后使用nn_layer定义神经网络输出层，其输入维度为上一层隐含节点数500，输出维度为类别数10，
# # 同时激活函数为全等映射identity，暂时不适用softmax
# #然后使用nn_layer定义神经网络输出层，其输入维度为上一层隐含节点数500，输出维度为类别数10，
# # 同时激活函数为全等映射identity，暂时不适用softmax
# # output2 =nn_layer(dropped,'layer2',act=tf.identity)#建立第二层 输出层
# #输出层（1个神经元）
# W2 = tf.Variable(tf.random_normal([10,1]))
# b2 = tf.Variable(tf.zeros([1,1])+0.1)
# Wx_plus_b2 = tf.matmul(hidden1,W2) + b2
# output2 = Wx_plus_b2
#
# #损失
# #使用tf.nn.softmax_cross_entropy_with_logits()对前面的输出层的结果进行softmax处理并计算
# # 交叉熵损失cross_entopy,计算平均损失，使用tf.summary.scalar进行统计汇总
# with tf.name_scope('cross_entropy'):
#     # diff=tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y)
#     loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - output2), reduction_indices=[1]))  # 在第一维上，偏差平方后求和，再求平均值，来计算损失
#     #输出层给出结果logits=y，每一行的y是有10个数预测10个值，然后利用这10个值做归一化，
#     # 然后具备一个概率的含义，第二不计算交叉熵
#     with tf.name_scope('total'):
#         cross_entropy=tf.reduce_mean(loss)#平均损失
# tf.summary.scalar('cross_entorpy',cross_entropy)
#
# # loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-output2),reduction_indices=[1])) #在第一维上，偏差平方后求和，再求平均值，来计算损失
# #下面使用Adam优化器对损失进行优化，同时统计预测正确的样本数并计算正确率accuracy，汇总
# with tf.name_scope('train'):
#     train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) # 使用梯度下降法，设置步长0.1，来最小化损失
#
# #Adamoptimizer比SGD更好一些，下降速度更快，更容易计算局部最优解，当数据量大的时候不如SGD
# #learning_rate虽然是固定的，后面会自适应，根据上一次的结果，所以大数据量的话，不如定义好策略，这样省时间
# with tf.name_scope('accuracy'):
#     with tf.name_scope('correct_predicition'):
#         correct_predition=tf.equal(tf.argmax(y,1),tf.argmax(output2,1))
#         #预测值最大的索引和真实值的索引
#     with tf.name_scope('accuracy'):
#         accuracy=tf.reduce_mean(tf.cast(correct_predition,tf.float32))
#         #true 1 false 0 reduce_mean 是一个比例得到的结果
# tf.summary.scalar('accuracy',accuracy)
#
# #因为我们之前定义了太多的tf.summary汇总操作，注意执行这些操作太
# # 麻烦，使用tf.summary.merge_all()直接获取所有汇总操作，以便后面执行
# merged=tf.summary.merge_all()
# #定义两个tf.summary.FileWirter文件记录器在不同的子目录，分别用来存储训练和测试的日志数据
# train_writer=tf.summary.FileWriter(log_dir+'/train',sess.graph)
# test_writer=tf.summary.FileWriter(log_dir+'/test')
# #同时，将Session计算图sess.graph加入训练过程，这样在TensorBoard的GRAPHS窗口中就能展示
# # 整个计算图的可视化效果，最后初始化全部变量
# tf.global_variables_initializer().run()
#
# #初始化
# init = tf.global_variables_initializer() #初始化所有变量
# sess = tf.Session()
# sess.run(init) #变量初始化
#
#
# # #定义feed_dict函数，如果是训练，需要设置dropout，如果是测试，keep_prob设置为1
# # def feed_dict(train):
# #     if train:#如果是训练数据的话需要droupout，测试的时候不要Droupout
# #         xs,ys= mnist.train.next_batch(100) #每一次拿一批次数据去训练
# #         k=dropout
# #     else:
# #         xs,ys=mnist.test.images,mnist.test.labels #真正测试的话全部测试，不是拿一批次的数据了
# #         k=1.0
# #     return {x:xs,y_:ys,keep_prob:k}
#
# #执行训练、测试、日志记录操作。创建模型的保存器
# saver=tf.train.Saver()
# #训练
# for i in range(max_steps): #训练1000次
#     # _,loss_value = sess.run([train_step,loss],feed_dict={x:x_data,y:y_data}) #进行梯度下降运算，并计算每一步的损失
#     if(i%50==0):
#         summary, loss_value = sess.run([merged, accuracy], feed_dict={x:x_data,y:y_data})
#         test_writer.add_summary(summary, i)  # 然后写出
#         print('Accuracy at step %s:%s' % (i, loss_value))
#         print(loss_value) # 每50步输出一次损失
#
# train_writer.close()
# test_writer.close()