#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class BaseEstimator(object):
    """base estimator for lipreading

    Args:
        model_parms: Dict. parameters to build model_fn
        run_config: RunConfig. config for `Estimator`
    """

    def __init__(self, model_parms,run_config):
        super(BaseEstimator, self).__init__()
        self.model_parms = model_parms
        self.run_config = run_config
        self.estimator = tf.estimator.Estimator(
            self.model_fn, params=self.model_parms,config=self.run_config)

    def train_and_evaluate(self,
                           train_input_fn,
                           eval_input_fn,
                           max_steps=1000000,
                           eval_steps=100,
                           throttle_secs=200):
        """train and eval.

        Args:
            train_input_fn: Input fn for Train.
            eval_input_fn: Input fn for Evaluation.

        Kwargs:
            max_steps: Max training steps.
            eval_steps: Steps to evaluate.
            throttle_secs: Evaluate interval. evaluation will perform only when new checkpoints exists.

        """
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=max_steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            throttle_secs=throttle_secs,
            steps=eval_steps)

        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

    def evaluate(self, eval_input_fn, steps=None, checkpoint_path=None):
        """evaluate and print

        Args:
            eval_input_fn: Input function.

        Kwargs:
            steps: Evaluate steps
            checkpoint_path: Checkpoint to evaluate.

        Returns: Evaluate results.

        """
        return self.estimator.evaluate(
            eval_input_fn, steps=steps, checkpoint_path=checkpoint_path)

    def predict(self, predict_input_fn, checkpoint_path=None):
        """predict new examples
        Args:
            predict_input_fn: Input fn.

        """
        predictions = self.estimator.predict(
            predict_input_fn, checkpoint_path=checkpoint_path)

        a=[]
        for i,prediction in enumerate(predictions):
            #print(prediction)
            a.append(prediction)
        np.save('predict.npy',np.array(a))

    def model_fn(self, features, labels, mode, params):

        raise NotImplementedError('model function is not implemented')
    
        
    @staticmethod
    def get_runConfig(model_dir,
                      save_checkpoints_steps,
                      multi_gpu = False,
                      keep_checkpoint_max=100):
        """ get RunConfig for Estimator.
        Args:
            model_dir: The directory to save and load checkpoints.
            save_checkpoints_steps: Step intervals to save checkpoints.
            keep_checkpoint_max: The max checkpoints to keep.
        Returns: Runconfig.

        """
        sess_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        if multi_gpu:
            distribution = tf.contrib.distribute.MirroredStrategy()
        else:
            distribution = None
        return tf.estimator.RunConfig(
            model_dir=model_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            keep_checkpoint_max=100,
            train_distribute=distribution,
            session_config=sess_config)
