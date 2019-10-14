import tensorflow as tf  


def residual_block(input_tensor, size, rate, dim, is_training=True): 
    with tf.variable_scope('fliter%d'%rate):
        conv_filter = aconv1d_layer(input_tensor, size=size, rate=rate, activation='tanh', is_training=is_training)
    with tf.variable_scope('gate%d'%rate):
        conv_gate = aconv1d_layer(input_tensor, size=size, rate=rate, activation='sigmoid', is_training=is_training)
    with tf.variable_scope('conv%d'%rate):
        out = conv_filter * conv_gate
        out = conv1d_layer(out, size=1, dim=dim)
    return out + input_tensor, out

def conv1d_layer(input_tensor, size=1, dim=128, bias=False, activation='tanh', is_training=True):
    shape = input_tensor.get_shape().as_list()
    out = tf.layers.conv1d(input_tensor, dim, size, use_bias=bias, activation=activation_wrapper(activation), padding="same")
    if not bias:
        out = tf.layers.batch_normalization(out, training=is_training)
    return out
def aconv1d_layer(input_tensor, size=7, rate=2, bias=False, activation='tanh', is_training=True):
    shape = input_tensor.get_shape().as_list()
    out = tf.layers.conv1d(input_tensor, shape[-1], size,  dilation_rate=rate, use_bias=bias, activation=activation_wrapper(activation), padding="same")
    if not bias:
        out = tf.layers.batch_normalization(out, training=is_training)      
    return out
def activation_wrapper(activation):
    if activation == 'sigmoid':
        return tf.nn.sigmoid
    elif activation == 'tanh':
        return tf.nn.tanh
    elif activation == 'elu':
        return tf.nn.elu 

def conv1d_bn(x, filters, kernel=5, activation=tf.nn.elu, rate=1, 
    padding='same', strides=1, use_bias=False, is_training=True):
    x = tf.layers.conv1d(x, filters, kernel, padding="same", dilation_rate=rate)
    x = tf.layers.batch_normalization(x, training=is_training)
    if activation!=None:
        x = activation(x)
    return x
def block_a(x, is_training=True):
    branch_0 = conv1d_bn(x, 96, 1, is_training=is_training)

    branch_1 = conv1d_bn(x, 64, 1, is_training=is_training)
    branch_1 = conv1d_bn(branch_1, 96, 5, is_training=is_training)

    branch_2 = conv1d_bn(x, 64, 1, is_training=is_training)
    branch_2 = conv1d_bn(branch_2, 96, 5, is_training=is_training)
    branch_2 = conv1d_bn(branch_2, 96, 5, is_training=is_training)

    branch_3 = tf.layers.average_pooling1d(x, 3, strides=1, padding='same')
    branch_3 = conv1d_bn(branch_3, 96, 1, is_training=is_training)

    x = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
    return x
def block_b(x, rate=1, is_training=True):

    branch_0 = conv1d_bn(x, 284, 1, is_training=is_training)

    branch_1 = conv1d_bn(x, 100, 1, is_training=is_training)
    branch_1 = conv1d_bn(branch_1, 256, 7, rate=rate, is_training=is_training)

    branch_2 = conv1d_bn(x, 100, 1, is_training=is_training)
    branch_2 = conv1d_bn(branch_2, 100, 7, rate=rate, is_training=is_training)
    branch_2 = conv1d_bn(branch_2, 124, 7, rate=rate, is_training=is_training)

    branch_3 = tf.layers.average_pooling1d(x, 7, strides=1, padding='same')
    branch_3 = conv1d_bn(branch_3, 90, 1, is_training=is_training)

    x = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
    return x
def block_c(self, x, rate=1):
    branch_0 = conv1d_bn(x, 284, 1, is_training=is_training)

    branch_1 = conv1d_bn(x, 100, 1, is_training=is_training)
    branch_1 = conv1d_bn(branch_1, 256, 7, rate=rate, is_training=is_training)

    branch_2 = conv1d_bn(x, 100, 1)
    branch_2 = conv1d_bn(branch_2, 100, 7, rate=rate*2, is_training=is_training)
    branch_2 = conv1d_bn(branch_2, 124, 7, rate=rate*4, is_training=is_training)

    branch_3 = tf.layers.average_pooling1d(x, 7, strides=1, padding='same')
    branch_3 = conv1d_bn(branch_3, 90, 1, is_training=is_training)

    x = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
    return x
def wavenet(input_data, n_dim=128, n_blocks=3, is_training=True):
    with tf.variable_scope('conv_layer1'):
        net = conv1d_layer(input_data, dim=n_dim)
        skip = 0
        with tf.variable_scope('ResNet'):
            for itr_nb in range(n_blocks):
                with tf.variable_scope('ResNet%d'%itr_nb):
                    for r in [2, 4, 8, 16, 32]:
                        net, s = residual_block(net, size=5, rate=r, dim=n_dim, is_training=is_training)
                        skip += s
        return net 
def brnn(input_data, is_training=True):
    net = tf.layers.conv1d(input_data, 32, 5, dilation_rate=1, activation=tf.nn.leaky_relu, padding="same") 
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv1d(net, 64, 5, dilation_rate=2, activation=tf.nn.leaky_relu, padding="same") 
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv1d(net, 128, 5, dilation_rate=4, activation=tf.nn.leaky_relu, padding="same") 
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv1d(net, 256, 5, dilation_rate=8, activation=tf.nn.leaky_relu, padding="same")  
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(256) for itr in range(2)])
    cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(256) for itr in range(2)]) 
    (out_fw, out_bw), (st_fw, st_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, net, dtype=tf.float32)
    net = tf.concat([out_fw, out_bw], axis=2)
    return net 
def inception(input_data, is_training=True):
    net = conv1d_bn(input_data, 32, is_training=is_training) 
    net = conv1d_bn(net, 64, is_training=is_training) 
    net = conv1d_bn(net, 128, is_training=is_training)  
    for itr in range(2):
        net = block_a(net, is_training=is_training)
    for itr in range(5):
        net = block_b(net, rate=2**itr, is_training=is_training)
    for itr in range(2):
        net = block_b(net, rate=2**(itr+5), is_training=is_training)  
    return net 
def unet(input_data, is_training=True):
    # 编码器
    net = tf.layers.conv1d(input_data, 32, 3, strides=1, activation=tf.nn.elu, padding="same") 
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv1d(net, 32, 3, strides=1, activation=tf.nn.elu, padding="same") 
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.max_pooling1d(net, 2, 2) 
    net = tf.layers.conv1d(net, 64, 3, strides=1, activation=tf.nn.elu, padding="same") 
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv1d(net, 64, 3, strides=1, activation=tf.nn.elu, padding="same") 
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.max_pooling1d(net, 2, 2) 
    net = tf.layers.conv1d(net, 128, 3, strides=1, activation=tf.nn.elu, padding="same") 
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv1d(net, 128, 3, strides=1, activation=tf.nn.elu, padding="same") 
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.max_pooling1d(net, 2, 2) 
    net = tf.layers.conv1d(net, 256, 3, strides=1, activation=tf.nn.elu, padding="same") 
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv1d(net, 256, 3, strides=1, activation=tf.nn.elu, padding="same") 
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    # 解码器
    net = tf.expand_dims(net, 1)
    net = tf.layers.conv2d_transpose(net, 256, 3, strides=(1, 2), activation=tf.nn.elu, padding="same") 
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv2d_transpose(net, 128, 3, strides=(1, 1), activation=tf.nn.elu, padding="same")
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv2d_transpose(net, 128, 3, strides=(1, 1), activation=tf.nn.elu, padding="same")
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv2d_transpose(net, 128, 3, strides=(1, 2), activation=tf.nn.elu, padding="same")
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv2d_transpose(net, 64, 3, strides=(1, 1), activation=tf.nn.elu, padding="same")
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv2d_transpose(net, 64, 3, strides=(1, 1), activation=tf.nn.elu, padding="same")
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv2d_transpose(net, 64, 3, strides=(1, 2), activation=tf.nn.elu, padding="same")
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv2d_transpose(net, 32, 3, strides=(1, 1), activation=tf.nn.elu, padding="same")
    net = tf.layers.batch_normalization(net, training=is_training, scale=False)
    net = tf.layers.conv2d_transpose(net, 32, 3, strides=(1, 1), activation=tf.nn.elu, padding="same")
    net = tf.squeeze(net)
    return net 
if __name__ == "__main__":
    inputs = tf.placeholder(tf.float32, [32, None, 3])
    net = unet_model(inputs)