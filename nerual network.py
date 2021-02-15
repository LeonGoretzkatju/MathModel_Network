import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


alpha = 0.01 # 学习率 alpha
epoch = 500 # 训练全量数据集的轮数
df1 = pd.read_excel("E:\project\data.xlsx",names=["lat","lon","mon","tem","water","num"])
# 定义归一化函数
def normalize_feature(df):
    return df.apply(lambda column:(column-column.mean()) / column.std())

# 重新查看数据
df = normalize_feature(df1)
df.head()
ones = pd.DataFrame({"ones":np.ones(len(df))})
df = pd.concat([ones,df],axis=1)
df.head()
X_data = np.array(df[df.columns[0:6]])
y_data = np.array(df[df.columns[-1]]).reshape(len(df),1)

print(X_data.shape,type(X_data))
print(y_data.shape,type(y_data))
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, X_data.shape, name='X')
    y = tf.placeholder(tf.float32, y_data.shape, name='y')

with tf.name_scope('hypothesis'):
    W = tf.get_variable("weights",
                        (X_data.shape[1], 1),
                        initializer=tf.constant_initializer())
    y_pred = tf.matmul(X, W, name='y_pred')

with tf.name_scope('loss'):
    loss_op = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)
with tf.name_scope('train'):
    # 随机梯度下降优化器 opt
    train_op = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss_op)

with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./summary/linear-regression-1', sess.graph)
    # 记录所有损失值
    loss_data = []
    # 开始训练模型
    # 因为训练集较小，所以采用批梯度下降优化算法，每次都使用全量数据训练
    for e in range(1, epoch + 1):
        _, loss, w = sess.run([train_op, loss_op, W], feed_dict={X: X_data, y: y_data})
        # 记录每一轮损失值变化情况
        loss_data.append(float(loss))
        if e % 10 == 0:
            log_str = "Epoch %d \t Loss=%.4g \t Model: y = %.4gx1 + %.4gx2 + %.4gx3 + %.4gx4 + %.4gx5 +%.4g "
            print(log_str % (e, loss, w[1], w[2],w[3],w[4],w[5],w[0]))

writer.close()
sns.set(context="notebook", style="whitegrid", palette="dark")

ax = sns.residplot(x='epoch', y='loss', data=pd.DataFrame({'loss': loss_data, 'epoch': np.arange(epoch)}))
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
plt.show()


