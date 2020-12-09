import tensorflow as tf
import tensorflow.keras as keras

class motionNet(object):

	def __init__(self):
		pass

	def motionExtractor(self,video):
		with tf.variable_scope('motion',reuse=tf.AUTO_REUSE):
			video=self.data_concat(video)
			

			self.conv1 = keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3,3), strides=(1,1),padding='same'))(video)
			self.pool1 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))(self.conv1)


			self.conv2 = keras.layers.TimeDistributed(keras.layers.Conv2D(96, (3,3), strides=(1,1),padding='same'))(self.pool1)
			self.pool2 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))(self.conv2)

			self.conv3=keras.layers.TimeDistributed(keras.layers.Conv2D(128, (3,3), strides=(1,1),padding='same'))(self.pool2)
			self.pool3 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))(self.conv3)

			self.conv4=keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3,3), strides=(1,1),padding='same'))(self.pool3)

			self.deconv3 = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(128, (5,5), strides=(2,2),padding='same'))(self.conv4)
			self.deconv2 = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(96, (5,5), strides=(2,2),padding='same'))(self.deconv3)
			self.deconv1 = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2),padding='same'))(self.deconv2)
		
			self.pred4 = keras.layers.TimeDistributed(keras.layers.Conv2D(2, (3,3), strides=(1,1),padding='same'))(self.conv4)

			self.flow4 = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(2, (5,5), strides=(2,2),padding='same'))(self.pred4)
			self.concat3=tf.concat([self.conv3,self.deconv3,self.flow4],-1)
			self.pred3 = keras.layers.TimeDistributed(keras.layers.Conv2D(2, (3,3), strides=(1,1),padding='same'))(self.concat3)

			self.flow3 = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(2, (5,5), strides=(2,2),padding='same'))(self.pred3)
			self.concat2=tf.concat([self.conv2,self.deconv2,self.flow3],-1)
			self.pred2 = keras.layers.TimeDistributed(keras.layers.Conv2D(2, (3,3), strides=(1,1),padding='same'))(self.concat2)

			self.flow2 = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(2, (5,5), strides=(2,2),padding='same'))(self.pred2)
			self.concat1=tf.concat([self.conv1,self.deconv1,self.flow2],-1)
			self.pred1 = keras.layers.TimeDistributed(keras.layers.Conv2D(2, (3,3), strides=(1,1),padding='same'))(self.concat1)

		return self.pred1,self.pred2,self.pred3,self.pred4

	def data_concat(self,video):
		concat_video=tf.expand_dims(tf.concat([video[:,0,::],video[:,1,::]],-1),1)

		for i in range(1,15):
			image_couple=tf.expand_dims(tf.concat([video[:,i,::],video[:,i+1,::]],-1),1)
			concat_video=tf.concat([concat_video,image_couple],1)
		
		return concat_video
