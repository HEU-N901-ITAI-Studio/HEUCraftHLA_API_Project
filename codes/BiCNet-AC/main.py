import gym.core
import numpy as np
import scipy as sp
import sympy as sy

import time
import tqdm

import gym
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, LSTM, Bidirectional, Conv1D
from tensorflow.keras import Model

# 状态空间分为共享空间与智能体局部状态空间

# 智能体局部状态空间包括针对地图几何中心点的距离和角度,
# 自身的生命值、攻击的冷却时间,以及周围若干个我方和敌方智能体的距离和角度
#


######################################################################################
# 奖励函数实现
# 分别对每个智能体的奖励进行计算,奖励值与距离最近的K个智能体相关,
# 其值等于K个智能体中敌方智能体生命值减少的平均值减去我方智能体生命值的平均值
# 该值经过了归一化
######################################################################################

#函数:get_distance
#功能:计算两个点距离
def get_distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

#函数:unit_top_k_reward
#功能:计算当前单位奖励函数
#输入:k(智能体个数),unit(当前单位),unit_dict_list(单位字典)
#    distance_list(当前单位与其他单位的距离),flag_list(单位身份)
#    health_delta_list(单位生命值变化)
#输出:unit_reward(归一化后的奖励值)
def unit_top_k_reward(k, unit, unit_dict_list):
    """
    计算单位的 top-k 奖励值，用于评估单位与其他目标之间的交互表现。

    参数：
    - k: 选取的 top-k 距离最近目标数量。
    - unit: 当前计算的单位对象，需包含 x 和 y 坐标。
    - unit_dict_list: 单位字典列表，每个字典包含单位的信息（例如位置、健康变化、死亡状态等）。

    返回：
    - unit_reward: 该单位的奖励值，基于 top-k 敌方和友方目标的健康变化与距离。

    [分析]
    - 基于单位与其他单位的距离和健康变化来计算奖励。
    - 敌方目标对奖励值产生正贡献，友方目标对奖励值产生负贡献。
    """
    distance_list = []  # 存储单位到其他目标的距离
    flag_list = []  # 存储单位阵营标志（敌方或友方）
    health_delta_list = []  # 存储目标的健康变化归一化值

    # 遍历单位字典列表，提取相关信息
    for unit_dict in unit_dict_list:
        flag = unit_dict.flag  # 当前单位的阵营标志（0 表示友方，1 表示敌方）
        for i in unit_dict.id_list:
            t = unit_dict.units_dict[i]  # 获取目标单位信息
            if t.die and t.delta_health <= 0:  # 忽略已死亡且健康变化为负的目标
                continue
            # 计算单位之间的距离
            d = get_distance(unit.x, unit.y, t.x, t.y)
            # 归一化健康变化（变化量/最大健康值）
            health_delta_list.append(t.delta_health * 100. / t.max_health)
            distance_list.append(d)
            flag_list.append(flag)

    # 找到 top-k 最近目标的索引
    top_k_indexes = np.argsort(np.array(distance_list))[:k]
    # 获取 top-k 最近目标的阵营标志
    top_k_flags = np.array(flag_list)[top_k_indexes]
    # 计算 top-k 中敌方和友方目标的数量
    num_enemy = np.sum(top_k_flags) + 0.  # 敌方目标数
    num_myself = len(top_k_indexes) - num_enemy + 0.  # 友方目标数
    # 获取 top-k 最近目标的健康变化
    top_k_delta = np.array(health_delta_list)[top_k_indexes]

    # 初始化友方和敌方健康变化归一化值
    myself_delta_norm = 0
    enemy_delta_norm = 0
    if num_myself > 0:  # 计算友方目标的健康变化归一化值
        myself_delta_norm = np.sum(np.multiply(top_k_delta, 1 - top_k_flags)) / num_myself
    if num_enemy > 0:  # 计算敌方目标的健康变化归一化值
        enemy_delta_norm = np.sum(np.multiply(top_k_delta, top_k_flags)) / num_enemy

    # 计算单位的奖励值（敌方贡献减去友方损失）
    unit_reward = enemy_delta_norm - myself_delta_norm
    return unit_reward



######################################################################################
# AC网络实现
# 
# 
# 
######################################################################################
#以下为使用TensorFlow1.x版本的代码,已弃用
'''
class Dynamic_Actor(Model):
    def __init__(self,nb_unit_actions,name='actor',layer_norm=True,\
                 time_step=5):
        super(Dynamic_Actor,self).__init__(name=name)
        self.nb_unit_actions=nb_unit_actions
        self.layer_norm=layer_norm
        self.time_step=time_step
    #au alive units
    def __call__(self,ud,mask,au,n_hidden=64,reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            #x[batch_size,time_step,obs_dim]
            x=ud
            if self.layer_norm:
                x=tc.layers.layer_norm
            x=tf.layers.dense(x,64)
            if self.layer_norm:
                x=tc.layers.layer_norm
            x=tf.nn.relu(x)
            shape=x.get_shape().as_list()
            x=tf.reshape(x,[-1,self.time_step,shape[-1]])
            #build bidirection lstm
            lstm_fw_cell=rnn.BasicSTMCell(n_hidden,forget_bias=1.0)
            lstm_bw_cell=rnn.BasicSTMCell(n_hidden,forget_bias=1.0)
            x,_=tf.nn.bidrectional_dynamic_rnn(lstm_fw_cell,1,lstm_bw_cell,x\
                                            dtype=tf.float32,sequence_length=au)
            x=tf.concat(x,2)
            if self.layer_norm:
                x=tc.layers.layer_norm(x,center=True,scale=True)
            x=tf.nn.relu(x)
            x=tf.reshape(x,[-1,self.time_step*n_hidden*2,1])
            x=tf.layers.convld(x,self.nb_unit_actions,\
                            kernel_size=n_hidden*2,\
                            strides=n_hidden*2,\
                            kernel_initializer=\
                                tf.random_uniform_initializer(\
                                    minval=-3e-3,maxval=3e-3\
                                )\
                            )
            x=tf.nn.tanh(x)
        return x
'''
#Actor网络实现
class Dynamic_Actor(Model):  # [TensorFlow=2.l2] 动态 Actor 模型
    def __init__(self, nb_unit_actions, name='actor', layer_norm=True, time_step=5):
        """
        初始化 Dynamic_Actor 类，定义其主要参数。

        参数：
        - nb_unit_actions: 动作空间的维度（例如一个多动作输出的维度）。
        - name: 模型的名称，默认为 'actor'。
        - layer_norm: 是否使用层归一化（Layer Normalization），布尔值。
        - time_step: 时间步数，用于处理序列输入，默认为 5。
        """
        super(Dynamic_Actor, self).__init__(name=name)  # [TensorFlow=2.l2] 调用父类的初始化方法
        self.nb_unit_actions = nb_unit_actions  # 动作空间的维度
        self.layer_norm = layer_norm  # 是否启用层归一化
        self.time_step = time_step  # 时间步数
        self.dense1 = Dense(64)  # [TensorFlow=2.l2] 全连接层
        self.ln1 = LayerNormalization() if self.layer_norm else None  # [TensorFlow=2.l2] 层归一化
        self.ln2 = LayerNormalization() if self.layer_norm else None
        self.bidirectional_lstm = Bidirectional(
            LSTM(64, return_sequences=True, return_state=False)
        )  # [TensorFlow=2.l2] 双向 LSTM
        self.ln3 = LayerNormalization(center=True, scale=True) if self.layer_norm else None
        self.conv1d = Conv1D(
            filters=self.nb_unit_actions,  # 输出通道数
            kernel_size=64 * 2,  # 卷积核大小
            strides=64 * 2,  # 步幅
            kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        )  # [TensorFlow=2.l2] 1D 卷积层

    def call(self, ud, mask, au):
        """
        定义模型的前向传播逻辑。

        参数：
        - ud: 输入数据张量，形状为 [batch_size, time_step, obs_dim]。
        - mask: 用于标记有效时间步的掩码张量（未使用）。
        - au: 序列的有效长度，用于动态 RNN。

        返回：
        - x: 处理后的动作输出张量。
        """
        x = ud  # 输入数据 `x`：形状为 [batch_size, time_step, obs_dim]
        # 全连接层 [TensorFlow=2.l2]
        x = self.dense1(x)
        # 层归一化（如果启用） [TensorFlow=2.l2]
        if self.layer_norm:
            x = self.ln1(x)
        # 使用 ReLU 激活函数 [TensorFlow=2.l2]
        x = tf.nn.relu(x)
        # 调整形状为 3D 张量：[-1, time_step, feature_dim]
        x = tf.reshape(x, [-1, self.time_step, x.shape[-1]])
        # 双向 LSTM [TensorFlow=2.l2]
        x = self.bidirectional_lstm(x, mask=tf.sequence_mask(au))
        # 层归一化（如果启用）
        if self.layer_norm:
            x = self.ln3(x)
        # 使用 ReLU 激活函数
        x = tf.nn.relu(x)
        # 调整形状：[-1, time_step * feature_dim, 1]
        x = tf.reshape(x, [-1, self.time_step * 64 * 2, 1])
        # 1D 卷积层 [TensorFlow=2.l2]
        x = self.conv1d(x)
        # 使用 tanh 激活函数
        x = tf.nn.tanh(x)
        return x
    
#以下为使用TensorFlow1.x版本的代码,已弃用
'''
class Dynamic_Critic(Model):
    def __init__(self,name='critic',layer_norm=True,time_step=5):
        super(Dynamic_Critic,self).__init__(name==name)
        self.layer_norm=layer_norm
        self.time_step=time_step
    def __call__(self,ud,action,mask,au,n_hidden=64,reuse=False,unit_data=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            #x[batch_size,time_step,obs_dim]
            x=ud
            if self.layer_norm:
                x=tc.layers.layer_norm(x,center=True,scale=True)
            x-tf.layers.dense(x,64)
            if self.layer_norm:
                x=tc.layers.layer_norm(x,center=True,scale=True)
            x=tf.nn.relu(x)
            #format action to be[batch_size*time_step,nb_actions]
            x=tf.concat([x,action],axis=-1)
            x=tf.layers.dense(x,64)
            if self.layer_norm:
                x=tc.layers.layer_norm(x,center=True,scale=True)
            x=tf.nn.relu(x)
            shape=x.get_shape().as_list()
            x=tf.reshape(x,[-1,self.time_step,shape[-1]])
            lstm_fw_cell=rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
            lstm_bw_cell=rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
            x,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,x,\
                                                dtype=tf.float32,sequence_length=au)
            x=tf.concat(x,2)
            if self.layer_norm:
                x=tc.layers.layer_norm(x,center=True,scale=True)
            x=tf.nn.relu(x)
            x=tf.reshape(x,[-1,self.time_step*n_hidden*2,1])
            q=tf.layers.conv1d(x,1,kernal_size=n_hidden*2,strides=n_hidden*2,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,maxval=3e-3))
            q=tf.squeeze(q,[-1])
            Q=tf.reduce_sum(q,axis=1,keepdims=True)
            if unit_data:
                return Q,q
            return Q
'''

# Critic 网络实现
class Dynamic_Critic(Model):  # [TensorFlow=2.l2] 动态 Critic 模型
    def __init__(self, name='critic', layer_norm=True, time_step=5):
        """
        初始化 Dynamic_Critic 类，定义其主要参数。

        参数：
        - name: 模型的名称，默认为 'critic'。
        - layer_norm: 是否使用层归一化（Layer Normalization），布尔值。
        - time_step: 时间步数，用于处理序列输入，默认为 5。
        """
        super(Dynamic_Critic, self).__init__(name=name)  # 调用父类的初始化方法
        self.layer_norm = layer_norm  # 是否启用层归一化
        self.time_step = time_step  # 时间步数
        self.dense1 = Dense(64)  # 全连接层
        self.ln1 = LayerNormalization() if self.layer_norm else None  # 层归一化
        self.ln2 = LayerNormalization() if self.layer_norm else None
        self.bidirectional_lstm = Bidirectional(
            LSTM(64, return_sequences=True, return_state=False)
        )  # 双向 LSTM
        self.ln3 = LayerNormalization(center=True, scale=True) if self.layer_norm else None
        self.conv1d = Conv1D(
            filters=1,  # 输出通道数为 1（计算 Q 值）
            kernel_size=64 * 2,  # 卷积核大小
            strides=64 * 2,  # 步幅
            kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        )  # 1D 卷积层

    def call(self, ud, action, mask, au, unit_data=False):
        """
        Critic 网络的前向传播逻辑。

        参数：
        - ud: 输入单元数据张量，形状为 [batch_size, time_step, obs_dim]。
        - action: 动作张量，形状为 [batch_size, action_dim]。
        - mask: 动态 RNN 掩码。
        - au: 动态 RNN 有效长度。
        - unit_data: 是否返回单个单元的 Q 值。

        返回：
        - Q: 整体 Q 值。
        - q: 单个单元的 Q 值（如果 unit_data=True）。
        """
        x = ud  # 输入的单元数据
        # 调整动作的形状以匹配时间维度
        action = tf.expand_dims(action, axis=1)  # (batch_size, 1, action_dim)
        action = tf.tile(action, [1, self.time_step, 1])  # (batch_size, time_step, action_dim)
        # 拼接单元数据和动作
        x = tf.concat([x, action], axis=-1)  # (batch_size, time_step, obs_dim + action_dim)
        # 全连接层
        x = self.dense1(x)
        # 层归一化（如果启用）
        if self.layer_norm:
            x = self.ln1(x)
        # 使用 ReLU 激活函数
        x = tf.nn.relu(x)
        # 双向 LSTM
        x = self.bidirectional_lstm(x, mask=tf.sequence_mask(au))
        # 层归一化（如果启用）
        if self.layer_norm:
            x = self.ln3(x)
        # 使用 ReLU 激活函数
        x = tf.nn.relu(x)
        # 调整形状为 3D 张量
        x = tf.reshape(x, [-1, self.time_step * 64 * 2, 1])
        # 1D 卷积层
        q = self.conv1d(x)
        # Q 值的计算
        q = tf.squeeze(q, axis=-1)  # (batch_size, num_units)
        Q = tf.reduce_sum(q, axis=1, keepdims=True)  # (batch_size, 1)
        if unit_data:
            return Q, q
        return Q

######################################################################################
# 训练算法实现
# 
# 
# 
######################################################################################
# 动作生成实现
def pi(self, obs, apply_noise=True, compute_Q=True):
    """
    基于当前的观察（obs）生成动作，并计算相应的 Q 值。

    参数：
    - obs: 当前的观察值，通常是一个字典，包含每个观察项的数据。
    - apply_noise: 是否应用噪声（如 Ornstein-Uhlenbeck 噪声）来增加探索性，默认为 True。
    - compute_Q: 是否计算 Q 值，默认为 True。

    返回：
    - action: 生成的动作。
    - q: 对应的 Q 值（如果 compute_Q=True）。
    - uq: 计算的单位 Q 值（如果 compute_Q=True）。
    """
    assert obs.keys() == self.obs.keys()  # 确保观察输入与模型的观察空间一致
    # 从网络中获取 actor 输出和 critic 相关的 TensorFlow 运算
    actor_tf = self.actor_tf  # 获取 actor 网络的操作
    # 创建 feed_dict，基于输入的观察字典构建 TensorFlow 的输入
    feed_dict = {self.objs0[k]: [obs[k]] for k in obs.keys()}  # 传入当前观察到的输入数据
    # 根据是否需要计算 Q 值，执行不同的操作
    if compute_Q:
        # 计算 actor 输出、critic 输出及单位 Q 值
        action, q, uq = self.sess.run([actor_tf, self.critic_with_actor_tf, self.uq_with_actor], feed_dict=feed_dict)
    else:
        # 仅计算动作，Q 值和单位 Q 值为 None
        action = self.sess.run(actor_tf, feed_dict=feed_dict)
        q = None
        uq = None
    # 去除维度为 1 的维度（通常是 batch_size 维度）
    action = np.squeeze(action, [0])
    # 如果定义了噪声，并且设置了 apply_noise 为 True，加入噪声
    if self.action_noise is not None and apply_noise:
        noise = self.action_noise()  # 获取噪声
        assert noise.shape == action.shape  # 确保噪声的形状与动作一致
        action += noise  # 将噪声加到动作上
    # 将生成的动作限制在预设的范围内
    action = np.clip(action, self.action_range[0], self.action_range[1])  # 动作限制在 [min, max] 范围内
    return action, q, uq  # 返回生成的动作和对应的 Q 值

# Ornstein-Uhlenbeck 噪声类
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        """
        初始化 Ornstein-Uhlenbeck 噪声。

        参数：
        - mu: 均值（长期目标值），通常为 0。
        - sigma: 噪声的标准差，控制噪声的强度。
        - theta: 反向速度，控制噪声的衰减速度。
        - dt: 时间步长。
        - x0: 初始状态，默认为 None。
        """
        self.theta = theta  # 反向速度，控制噪声的衰减
        self.mu = mu  # 均值（目标值）
        self.sigma = sigma  # 噪声的标准差
        self.dt = dt  # 时间步长
        self.x0 = x0  # 初始状态
        self.reset()  # 初始化噪声状态

    def __call__(self):
        """
        生成一个新的噪声值。

        返回：
        - x: 当前时间步的噪声。
        """
        # 计算 Ornstein-Uhlenbeck 噪声
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x  # 更新前一个状态
        return x  # 返回新的噪声值

#AC网络智能体训练函数
def train(self):
    """
    执行一次训练过程，包括从经验池中采样一个批次，计算目标 Q 值，并更新 Actor 和 Critic 网络的权重。

    返回：
    - critic_loss: Critic 网络的损失值。
    - actor_loss: Actor 网络的损失值。
    """
    # 从经验池中随机采样一个批次
    batch = self.memory.sample(batch_size=self.batch_size)
    # 构建目标数据字典，包含观察值、奖励和下一状态的终止标志
    target_feed_dict = {self.obs1[k]: batch["obs1"][k] for k in self.obs1.keys()}
    target_feed_dict.update({
        self.rewards: batch["rewards"],  # 当前批次的奖励值
        self.terminals1: batch["terminals1"].astype("float32")  # 下一状态是否终止
    })
    # 计算目标 Q 值
    target_uq = self.sess.run(self.target_uq, feed_dict=target_feed_dict)
    # 设置需要运行的操作（更新 Actor 和 Critic 网络的操作）
    ops = [self.actor_train_op, self.actor_loss, self.critic_train_op, self.critic_loss]
    # 构建训练数据字典，包含当前状态的观察值和动作
    feed_dict = {self.obs0[k]: batch["obs0"][k] for k in self.obs0.keys()}
    feed_dict.update({
        self.actions: batch["actions"],  # 当前批次的动作
        self.critic_target: target_uq  # 用计算得到的目标 Q 值作为 Critic 的目标
    })
    # 运行训练操作，返回 Actor 和 Critic 的损失值
    _, actor_loss, _, critic_loss = self.sess.run(ops, feed_dict=feed_dict)
    # 返回损失值
    return critic_loss, actor_loss

#Actor网络参数更新实现
def setup_actor_optimizer(self):
    """
    设置 Actor 网络的优化器，包括计算损失函数、梯度、以及梯度裁剪，并生成训练操作。

    主要功能：
    - 定义 Actor 网络的损失函数。
    - 计算 Actor 网络参数的梯度。
    - 选择性应用梯度裁剪。
    - 配置优化器并生成训练操作。
    """
    # 打印提示信息，表示正在设置 Actor 网络的优化器
    print("setting up actor optimizer")
    # 定义 Actor 网络的损失函数，使用负的 Critic 输出（目标是最大化 Critic 的输出）
    self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
    # 打印 Actor 网络中可训练变量的形状信息，用于调试和验证
    actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
    print("actor.shapes:{}".format(actor_shapes))
    # 创建一个独立的命名空间，便于区分优化器相关的变量
    with tf.variable_scope("actor_optimizer"):
        # 定义 Adam 优化器，用于优化 Actor 网络的参数
        self.actor_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.actor_lr,  # 学习率
            beta1=0.9,  # 一阶动量超参数
            beta2=0.0999,  # 二阶动量超参数
            epsilon=1e-8  # 防止除零的小值
        )
        # 计算 Actor 网络的梯度（基于损失函数对可训练变量的偏导数）
        actor_grads = tf.gradients(self.actor_loss, self.actor.trainable_vars)
        # 如果设置了梯度裁剪，应用全局梯度裁剪
        if self.clip_norm:
            # `tf.clip_by_global_norm` 用于限制梯度的全局范数，防止梯度爆炸
            self.actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.clip_norm)
        else:
            # 如果没有设置梯度裁剪，直接使用原始的梯度
            self.actor_grads = actor_grads
        # 将梯度和对应的变量组合成一个列表，用于优化器的输入
        grads_and_vars = list(zip(self.actor_grads, self.actor.trainable_vars))
        # 定义训练操作，应用计算得到的梯度以更新 Actor 网络的参数
        self.actor_train_op = self.actor_optimizer.apply_gradients(grads_and_vars)

#Critic网络参数更新实现
def setup_critic_optimizer(self):
    """
    设置 Critic 网络的优化器，计算损失、添加 L2 正则化项，并生成训练操作。

    主要功能：
    - 定义 Critic 网络的损失函数（与目标值的差的平方和）。
    - 如果启用了 L2 正则化，则添加正则化项。
    - 计算 Critic 网络参数的梯度。
    - 配置优化器并生成训练操作。
    """
    # 打印提示信息，表示正在设置 Critic 网络的优化器
    print("setting up critic optimizer")
    # 定义 Critic 网络的损失函数，计算目标值与 Critic 输出（预测值）之间的均方误差
    loss1 = tf.reduce_sum(tf.square(self.uq - self.critic_target), axis=1, keepdims=True)
    self.critic_loss = tf.reduce_mean(loss1)
    # 如果启用了 L2 正则化，则对网络的某些权重应用正则化
    if self.critic_l2_reg > 0.:
        # 找到需要进行 L2 正则化的变量，这里假设对 "kernel" 或 "W"（权重）进行正则化
        critic_reg_vars = [var for var in self.critic.trainable_vars if ("kernel" in var.name or "W" in var.name) and "output" not in var.name]
        # 对每个正则化的变量进行打印调试
        for var in critic_reg_vars:
            print("regularizing: {}".format(var.name))
            print("applying l2 regularization with {}".format(self.critic_l2_reg))
        # 为选定的变量应用 L2 正则化
        critic_reg = tc.layers.apply_regularization(
            tc.layers.l2_regularizer(self.critic_l2_reg),
            weights_list=critic_reg_vars
        )
        # 将正则化项加入 Critic 损失
        self.critic_loss += critic_reg
        # 打印 Critic 网络可训练变量的形状，便于调试
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        print("critic shapes:{}".format(critic_shapes))
    # 创建一个独立的命名空间，便于区分优化器相关的变量
    with tf.variable_scope("critic_optimizer"):
        # 定义 Adam 优化器，用于优化 Critic 网络的参数
        self.critic_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.critic_lr,  # 学习率
            beta1=0.9,  # 一阶动量超参数
            beta2=0.999,  # 二阶动量超参数
            epsilon=1e-8  # 防止除零的小值
        )
        # 计算 Critic 网络的梯度（基于损失函数对可训练变量的偏导数）
        critic_grads = tf.gradients(self.critic_loss, self.critic.trainable_vars)
        # 如果设置了梯度裁剪，应用全局梯度裁剪
        if self.clip_norm:
            # `tf.clip_by_global_norm` 用于限制梯度的全局范数，防止梯度爆炸
            self.critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.clip_norm)
        else:
            # 如果没有设置梯度裁剪，直接使用原始的梯度
            self.critic_grads = critic_grads
    # 将梯度和对应的变量组合成一个列表，用于优化器的输入
    grads_and_vars = list(zip(self.critic_grads, self.critic.trainable_vars))
    # 定义训练操作，应用计算得到的梯度以更新 Critic 网络的参数
    self.critic_train_op = self.critic_optimizer.apply_gradients(grads_and_vars)

#TODO: 加注释,单元测试
#目标网络参数更新
def update_target_net(self):
    self.sess.run(self.target_soft_updates)
#设置目标A-C网络参数的更新方法
def get_target_updates(vars,target_vars,tau):
    print("setting up target updates...")
    soft_updates=[]
    init_updates=[]
    for i in vars:
        print(i.name)
    for i in target_vars:
        print(i.name)
    print(len(vars),len(target_vars))
    assert len(vars)==len(target_vars)
    for var,target_var in zip(vars,target_vars):
        init_updates.append(tf.assign(target_vars))
        soft_updates.append(tf.assign(target_var,(1-tau)*target_var+tau*var))
    assert len(init_updates)==len(vars)
    assert len(soft_updates)==len(vars)
    return tf.group(*init_updates),tf.group(*soft_updates)
def setup_taget_network_updates(self):
    actor_init_updates,actor_soft_updates=get_target_updates(self.actor.vars,self.target_actor.vars,self.tau)
    critic_init_updates,critic_soft_updates=get_target_updates(self.critic.vars,self.target_critic.vars,self.tau)
    self.target_init_updates[actor_init_updates,critic_init_updates]
    self.target_soft_updates=[actor_soft_updates,critic_soft_updates]

#TODO:缺少数据输入与处理部分
######################################################################################
# 程序执行部分
# 
# 
# 
######################################################################################
env = gym.make("CartPole-v1")
#智能体训练过程
for epoch in range(nb_epochs):
    epoch_start_time=time.time()
    for cycle in tqdm(range(nb_epoch_cycles),ncols=50):
        while not done:
            action,q,uq=agent.pi(obs,apply_noise=True,compute_Q=True)
            assert action.shape==env.action_space.shape
            new_obs,r,done,info=env.step(action)
            t+=1
            episode_reward+=np.sum(r)
            episode_step+=1
            epoch_actions.append(action)
            epoch_qs.append(q)
            agent.store_transition(obs,action,r,new_obs,done)
            obs=new_obs
            if done:
                for _ in range(nb_train_steps):
                    if agent.memory.length>50*batch_size:
                        cl,al=agent.train()
                        epoch_critic_losses.append(cl)
                        epoch_actor_losses.append(al)
                        agent.update_target_net()
                    epoch_episode_steps.append(episode_step)
                    episode_reward=0.
                    episode_step=0
                    epoch_episodes+=1
                    episodes+=1
                    agent.reset()
                    obs=env.reset()
                done = False

#训练后加载模型参数,测试算法性能
with tf.Session() as sess:
    saver.restore(sess,vars_dict["model"])
    graph=tf.get_default_gradph()
    obs=get_obs_tensor(observation_dtype.keys(),"obs0",graph)
    env_obs=env.reset()
    done=Falseoutput=graph.get_tensor_by_name("actor/Tanh:0")
    assert(set(env_obs.keys())==set(obs.keys()))
    for cycle in tqdm(rang(total_games),ncols=50):
        while not done:
            feed_dict={obs[k]:[env_obs[k]]for k in env_obs.keys()}
        action=sess.run(output,feed_dict=feed_dict)
        action=np.squeeze(action,[0])
        env_obs,r,done,info=env.step(action)
        if done:
            if env.check_win():
                num_win+=1
            env_obs=env.reset()
        done=False
    print("win rate for %d games is %0.3f"%(total_games,num_win/total_games))
    print("cost time:",time.time()-begin_time)