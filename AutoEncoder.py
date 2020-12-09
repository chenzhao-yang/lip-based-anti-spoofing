import tensorflow as tf
import tensorflow.keras as keras


class CNN(object):

    def __init__(self, training, scope='cnn_feature_extractor',channel=2):
       
        self.training = training
        self.scope = scope
        self.channel = channel

    def encoder():
        raise NotImplementedError('CNN not NotImplemented.')
    def decoder():
        raise NotImplementedError('CNN not NotImplemented.')

class AutoEncoder(CNN):

    def __init__(self, *args, **kwargs):
        super(AutoEncoder, self).__init__(*args, **kwargs)

    def encoder(self, video_tensor):
    
        with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
            self.conv1 = keras.layers.Conv3D(32, (3,3,3), strides=(1,1,1),padding='same', name='conv1')(video_tensor)
            self.batc1 = tf.layers.batch_normalization(self.conv1, name= 'batc1', training=self.training)
            self.actv1 = keras.layers.Activation('relu', name='actv1')(self.batc1)
            self.drop1 = keras.layers.SpatialDropout3D(0)(self.actv1, training=self.training)
            self.maxp1 = keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),padding='same')(self.drop1)

            self.conv2 = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1),padding='same', name='conv2')(self.maxp1)
            self.batc2 = tf.layers.batch_normalization(self.conv2, name= 'batc2', training=self.training)
            self.actv2 = keras.layers.Activation('relu', name='actv2')(self.batc2)
            self.drop2 = keras.layers.SpatialDropout3D(0)(self.actv2, training=self.training)
            self.maxp2 = keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),padding='same')(self.drop2)
            
            self.conv3 = keras.layers.Conv3D(96, (3,3,3), strides=(1,1,1),padding='same', name='conv3')(self.maxp2)
            self.batc3 = tf.layers.batch_normalization(self.conv3, name= 'batc3', training=self.training)
            self.actv3 = keras.layers.Activation('relu', name='actv3')(self.batc3)
            self.drop3 = keras.layers.SpatialDropout3D(0)(self.actv3, training=self.training)
            self.maxp3 = keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same')(self.drop3)

            self.encoded = keras.layers.Conv3D(128, (1,1,1), strides=(1,1,1),padding='same')(self.maxp3)
            return self.encoded
    
    def decoder(self, encoded):
        with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
            self.deconv3 = keras.layers.Conv3D(96, (3,3,3), strides=(1,1,1), padding='same',name='deconv3')(encoded)
            self.debatc3 = tf.layers.batch_normalization(self.deconv3, name= 'debatc3', training=self.training)
            self.deactv3 = keras.layers.Activation('relu', name='deactv3')(self.debatc3)
            self.dedrop3 = keras.layers.SpatialDropout3D(0)(self.deactv3,training=self.training)
            self.up3 = keras.layers.UpSampling3D((1, 2, 2))(self.dedrop3)

            self.deconv2 = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', name='deconv2')(self.up3)
            self.debatc2 = tf.layers.batch_normalization(self.deconv2, name= 'debatc2', training=self.training)
            self.deactv2 = keras.layers.Activation('relu', name='deactv2')(self.debatc2)
            self.dedrop2 = keras.layers.SpatialDropout3D(0)(self.deactv2,training=self.training)
            self.up2 = keras.layers.UpSampling3D((1, 2, 2))(self.dedrop2)

            self.deconv1 = keras.layers.Conv3D(32, (3,3,3), strides=(1,1,1), padding='same', name='deconv1')(self.up2)
            self.debatc1 = tf.layers.batch_normalization(self.deconv1, name= 'debatc1', training=self.training)
            self.deactv1 = keras.layers.Activation('relu', name='deactv1')(self.debatc1)
            self.dedrop1 = keras.layers.SpatialDropout3D(0)(self.deactv1,training=self.training)
            self.up1 = keras.layers.UpSampling3D((1, 2, 2))(self.dedrop1)


            self.decoded = keras.layers.Conv3D(self.channel, (1,1,1),strides=(1,1,1), padding='same')(self.up1)
            return self.decoded
    
