import tensorflow as tf

def mynet(inp, reuse=False):
    with tf.variable_scope("model"):
        print(inp.shape)
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(inp, 32, 15, 
                                        stride=1,
                                        activation_fn=tf.nn.relu, 
                                        padding='VALID',
                                        weights_initializer=tf.keras.initializers.he_normal(),
                                        scope=scope,
                                        reuse=reuse)
            
            print(net.shape)
            
            net = tf.contrib.layers.max_pool2d(net, kernel_size=2, stride=2)
                                    
            net = tf.contrib.layers.batch_norm(net, reuse=reuse, scope=scope)
            
            print(net.shape)
            

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, 8,
                                        stride=1,
                                        activation_fn=tf.nn.relu, 
                                        padding='VALID',
                                        weights_initializer=tf.keras.initializers.he_normal(),
                                        scope=scope,
                                        reuse=reuse)
            
            print(net.shape)
            
            net = tf.contrib.layers.max_pool2d(net, kernel_size=3, stride=3)
                    
            net = tf.contrib.layers.batch_norm(net, reuse=reuse, scope=scope)
            
            print(net.shape)
        
        
        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 256, 5, 
                                        stride=1,
                                        activation_fn=tf.nn.relu,
                                        padding='VALID',
                                        weights_initializer=tf.keras.initializers.he_normal(),
                                        scope=scope,
                                        reuse=reuse)
            
            print(net.shape)
            
            net = tf.contrib.layers.max_pool2d(net, kernel_size=2, stride=2)

            net = tf.contrib.layers.batch_norm(net, reuse=reuse, scope=scope)
            
            print(net.shape)

        
        with tf.variable_scope("fc1") as scope:
            net = tf.contrib.layers.flatten(net)
            print(net.shape)
        
        with tf.variable_scope("fc2") as scope:
            net = tf.contrib.layers.fully_connected(net, 64, 
                                                    activation_fn=tf.nn.relu, 
                                                    reuse=reuse, 
                                                    scope=scope)
            print(net.shape)
        
        
    return net


def contrastive_loss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keepdims=True))
        tmp = y * tf.square(d)    
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
    return tf.reduce_mean(tmp + tmp2)/2