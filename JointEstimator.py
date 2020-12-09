import tensorflow as tf
import tensorflow.contrib.keras as keras
from .AutoEncoder import AutoEncoder
from .base_estimator import BaseEstimator
from .motionNet import motionNet

class JointEstimator(BaseEstimator):
   
    
    def __init__(self, model_parms,run_config):
        super(JointEstimator, self).__init__(model_parms,run_config)

    def model_fn(self, features, labels, mode, params):
        
        learning_rate = params.get('learning_rate', 0.0001)
        thre = params.get('thre', 0)
        self.classes = params.get('classes', 24)
        joint = params.get('joint', True)
        isMotion = params.get('isMotion', True)
        self.a = params.get('a', 2)
        self.b = params.get('b', 1)
        video = features['video']
        real_motion = features['motion']
        labels = labels['label']


        if isMotion:

            motionNets = motionNet()
            pred_motion, pred_motion2, pred_motion3, pred_motion4 = motionNets.motionExtractor(video)
            motion = pred_motion

        self.batch_size = tf.shape(video)[0]
        self.T_size = tf.shape(video)[1]
        self.height = tf.shape(video)[2]
        self.width = tf.shape(video)[3]

        in_training = mode == tf.estimator.ModeKeys.TRAIN
        channel = 2 if isMotion else 3
        feature_extractor =AutoEncoder(
            training=in_training,
            scope='AutoEncoder',
            channel=channel)

        encoded = feature_extractor.encoder(motion) if isMotion else feature_extractor.encoder(video)
        decoded = feature_extractor.decoder(encoded)

        logits = self.classifier(encoded)

        

        
        #predictions = self.pre_by_thre(logits, thre)
        predictions = tf.argmax(logits, -1)

                
        #predict
        if mode == tf.estimator.ModeKeys.PREDICT:
            
            #export
            
            predict_output = {
                'predictions':
                predictions,   
            }
            
            export_outputs = {
                'predictions':
                tf.estimator.export.PredictOutput(predict_output)
            }
            
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)
        

        acc = self.cal_acc(labels, predictions)
        Lc = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        #cal Lr
        Lr = tf.losses.mean_squared_error(motion, decoded) if isMotion else tf.losses.mean_squared_error(video, decoded)
        loss = Lc + self.b * Lr if joint else Lc

        #cal Lm
        if isMotion:
            Lm_1, Lm_2, Lm_3, Lm_4 = self.motionLoss(motion, pred_motion2, pred_motion3, pred_motion4, real_motion)
            loss += self.a * (Lm_1 + Lm_2 + Lm_3 + Lm_4)

        tf.summary.scalar('loss', loss)

        #train
        if mode == tf.estimator.ModeKeys.TRAIN:
            
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            global_step = tf.train.get_global_step()
            tvars = tf.trainable_variables()

            gradients = optimizer.compute_gradients(
            loss, tvars, colocate_gradients_with_ops=True)

            minimize_op = optimizer.apply_gradients(
            gradients, global_step=global_step, name="train")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group(minimize_op, update_ops)

            #results, avgFRR = self.cal_avgFRR(labels, predictions)
            logging_hook = tf.train.LoggingTensorHook(
             {
                'loss': loss,
                #'FRR': avgFRR,
                'acc': acc,
                #'results': results[:10],
                'predictions': predictions[:10],
                'labels': labels[:10]
             },
             every_n_iter=100)
            
            estimator_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[logging_hook]
                )
            return estimator_spec
       
        #eval
        if mode == tf.estimator.ModeKeys.EVAL:

            eval_metric_ops = {
                'acc': tf.metrics.mean(acc)
            }

            logging_hook = tf.train.LoggingTensorHook(
             {
                'loss': loss,
                'label': labels[:10],
                'predictions': predictions[:10],
                'acc': acc
             },
             every_n_iter=10)
            
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=[logging_hook]           
            )

    #get prediction by threshold
    def pre_by_thre(self, logits, thre):
        compare = tf.greater(logits, thre)
        predictions = tf.where(compare, tf.zeros(tf.shape(compare), dtype=tf.int64), tf.ones(tf.shape(compare), dtype=tf.int64))
        return predictions


    def classifier(self,encoded):
      with tf.variable_scope('classifier',reuse=tf.AUTO_REUSE):
        self.pool=keras.layers.GlobalAveragePooling3D()(encoded)
        self.flatten = keras.layers.Flatten()(self.pool)
        self.dense1= keras.layers.Dense(128)(self.flatten)
        self.output = keras.layers.Dense(self.classes)(self.dense1)

        return self.output #B*C



    def motionLoss(self,pred_motion,pred_motion2,pred_motion3,pred_motion4,real_motion):

        motionLoss1=self.average_endpoint_error(real_motion,pred_motion)

        real_motion2=tf.reshape(tf.image.resize_images(tf.reshape(real_motion,[-1,40,64,2]),[20,32]),[self.batch_size,self.T_size-1,20,32,2])
        motionLoss2=self.average_endpoint_error(real_motion2,pred_motion2)

        real_motion3=tf.reshape(tf.image.resize_images(tf.reshape(real_motion,[-1,40,64,2]),[10,16]),[self.batch_size,self.T_size-1,10,16,2])
        motionLoss3=self.average_endpoint_error(real_motion3,pred_motion3)

        real_motion4=tf.reshape(tf.image.resize_images(tf.reshape(real_motion,[-1,40,64,2]),[5,8]),[self.batch_size,self.T_size-1,5,8,2])
        motionLoss4=self.average_endpoint_error(real_motion4,pred_motion4)

        return motionLoss1, motionLoss2, motionLoss3, motionLoss4


    def average_endpoint_error(self,labels, predictions):
        predictions = tf.cast(predictions, dtype = tf.float32)
        labels = tf.cast(labels, dtype = tf.float32)
        squared_difference = tf.square(tf.abs(predictions-labels))
        loss = tf.reduce_sum(squared_difference, 4, keepdims=True)
        loss = tf.sqrt(loss+1e-8)
        return tf.reduce_mean(loss)

    def cal_acc(self,labels,predictions):
        '''
        Cal classifying accuracy
        '''
        isequal=tf.equal(labels,predictions)
        result=tf.cast(isequal,dtype=tf.float16)
        acc=tf.reduce_mean(result)
        return acc

    def cal_avgFRR(self, labels, predictions):
        labels = tf.expand_dims(labels, axis=-1)
        results = tf.batch_gather(predictions, labels)
        avgFRR = tf.reduce_mean(tf.cast(results, dtype=tf.float16))
        return results, avgFRR


    def cal_avgFARD(self, labels, predictions):
        labels = tf.expand_dims(labels, axis=-1)
        results = tf.batch_gather(predictions, labels)
        avgTRRD = tf.reduce_mean(tf.cast(results, dtype=tf.float16))
        avgFARD = 1 - avgTRRD
        return avgFARD




