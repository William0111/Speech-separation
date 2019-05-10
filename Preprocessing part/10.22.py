

#学习目的：review

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.2 + 0.7

# plt.plot(x_data, y_data)
# plt.show()

#Structure
Weights = tf.Variable(tf.random_uniform([1],-1,1))      #维度为1，【-1，1】随机均匀分布
Biases = tf.Variable(tf.zeros([1])) #维度为1，初始为0

y = Weights*x_data + Biases     #y=Wx+b

loss = tf.reduce_mean(tf.square(y-y_data))      #计算loss
optimizer = tf.train.GradientDescentOptimizer(0.4)      #优化器，最基本的GradientDescentOptimizer,每一步优化0.4
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()        #这一步必不可少
#Structure

sess = tf.Session()
sess.run(init)  #Dont forget this step，一定要run一下这个init


for step in range(100):     #这就开始训练了，训练100步，每10步打印出一次W & b, 记住打印的时候也要用sess.run()这就类似一个指针的作用
    sess.run(train)
    if step%10 == 0:
        print(step,sess.run(Weights),sess.run(Biases))

matrix1 = tf.constant([[4],[4]]) #两行一列的矩阵
matrix2 = tf.constant([[8,8]])   #一行两列的矩阵

product = tf.matmul(matrix2, matrix1) #矩阵相乘

#method 1
result = sess.run(product)
print(result)
sess.close()

#method 2
with tf.Session() as sesss:
    result2 = sesss.run(product)
    print(result2)

#矩阵相乘出结果的两种方法，第二种不需要close

state = tf.Variable(0,name='counter')   #计步器
#print(state.name)
one = tf.constant(1)    #张量里面的1

new_value = tf.add(state,one)       #往里面加一
update = tf.assign(state,new_value) #把new_value的值传递到update

init = tf.initialize_all_variables()

with tf.Session() as sess: #加了5次
    sess.run(init)
    for _ in range(9):
        sess.run(update)
        print(sess.run(state))


input1 = tf.placeholder(tf.float32) #float32是tf里面最普适的数据类型
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2) #数据乘法

with tf.Session() as sess:
     print(sess.run(output,feed_dict={input1:[8],input2:[3]}))
     #placeholder需要feed_dict给赋值


def add_layer(input,in_size,out_size, n_layer, activation_function=None):
    #define了一个add_layer,里面需要赋值的有，输入，输入的size，输出的size，layer的名字，激励函数是什么

    layer_name = 'layer%s'% n_layer

    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_uniform([in_size,out_size]), name='W')
            #W的size应当是in_size X out_size,这不难想,比如你认为要有多少行，显然矩阵要能相乘，必须要和in_size一致；
            # 而多少列就是你希望输出矩阵的多少列
            tf.summary.histogram(layer_name+'Weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1, name='b')
            #b是一个数值，每一条神经元链接的线上都有一个b,那么显然b的个数由输出的个数多少决定
            tf.summary.histogram(layer_name + 'biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input,Weights) + biases
            #矩阵相乘加上b
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name + 'outputs', outputs)
        return outputs

#人造数据
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.1,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.8 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1], name='x_input')
    ys = tf.placeholder(tf.float32,[None,1], name='y_input')
#可视化操作中的步骤

l1 = add_layer(xs,1,10, n_layer=111, activation_function=tf.nn.relu)
#第一层，输入为1，输出为10，relu作为激励函数
prediction = add_layer(l1,10,1, n_layer=222, activation_function=None)
#第二次，输入为10.输出为1，没有激励函数

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    #计算loss,非常关键的一部
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    #优化器是GD,0.1的学习率

init = tf.global_variables_initializer()

sess = tf.Session()
merge = tf.summary.merge_all()
#merge就相当于把上面可视化操作全合并在一起
writer = tf.summary.FileWriter('/Users/admin/Desktop/tensorgraph/',sess.graph)
#z这一步是把生成并存入（写入）指定文件夹中

sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
#plot至今没玩熟练，还是要花时间去系统学

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        result = sess.run(merge,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)

        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.plot(x_data,prediction_value)
        plt.pause(0.1)






