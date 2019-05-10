
#William 2018/10/20/16:51

#主要练习目的： 1，学习tensorflow

import tensorflow as tf
import numpy as np



x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.2 + 0.7


#Structure
Weights = tf.Variable(tf.random_uniform([1],-1,1))
Biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + Biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.4)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
#Structure

sess = tf.Session()
sess.run(init)  #Dont forget this step

for step in range(100):
    sess.run(train)
    if step%10 == 0:
        print(step,sess.run(Weights),sess.run(Biases))


matrix1 = tf.constant([[4],[4]])
matrix2 = tf.constant([[8,8]])

product = tf.matmul(matrix2, matrix1)


#method 1
result = sess.run(product)
print(result)
sess.close()

#method 2
with tf.Session() as sesss:
    result2 = sesss.run(product)
    print(result2)



state = tf.Variable(0,name='counter')
#print(state.name)
one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))

#placeholder需要feed_dict给赋值
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
     print(sess.run(output,feed_dict={input1:[9.],input2:[3]}))




















