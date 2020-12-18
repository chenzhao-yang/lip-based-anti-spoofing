import os
import argparse
import tensorflow as tf
from JointEstimator import JointEstimator
from dataset import tfrecord_input_fn

#CURRENT_FILE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='either train, eval, predict')
  
    # train
    parser.add_argument(
        '--save_steps',
        type=int,
        default=200,
        help='steps interval to save checkpoint')
    parser.add_argument('--model_dir', help='directory to save checkpoints')

    parser.add_argument(
        '--train_file',
        type=str,
        default='',
        help='train file path')

    # eval
    parser.add_argument(
        '--eval_steps', type=int, default=1000, help='steps to eval')
    parser.add_argument(
        '--eval_file',
        type=str,
        default='',
        help='eval file path')

    # eval and predict
    parser.add_argument(
        '--ckpt_path', help='checkpoints to evaluate/predict', default=None)


    # misc
    parser.add_argument('-gpu', '--gpu', help='gpu id to use', default='')
   
   
    return parser.parse_args()


def main():
    args = arg_parse()
    if args.gpu != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    tf.logging.set_verbosity(tf.logging.INFO)

    multi_gpu = len(args.gpu.split(',')) > 1
    # build estimator
    run_config = JointEstimator.get_runConfig(
        args.model_dir,
        args.save_steps,
        multi_gpu=multi_gpu,
        keep_checkpoint_max=100)

    model_parms = {
            'learning_rate': 0.0001,
            'classes':24,
            'thre':0,
            'a': 2,
            'b': 1,
            'joint':True,
            'isMotion':	True,
              }


    model = JointEstimator(model_parms, run_config)

    # build input

    train_file = args.train_file
    test_file = args.eval_file

    
    train_input_params = {
        'num_epochs': 1000,
        'batch_size': 50,
        'num_threads': 4,
        'file_name_pattern': train_file
    }
    eval_input_params = {
        'num_epochs': 1,
        'batch_size': 50,
        'num_threads': 4,
        'file_name_pattern': test_file
    }
    train_input_fn = lambda: tfrecord_input_fn(mode=tf.estimator.ModeKeys.TRAIN, **train_input_params)
    eval_input_fn = lambda: tfrecord_input_fn(mode=tf.estimator.ModeKeys.EVAL, **eval_input_params)

    #begin train,eval,predict
    if args.mode == 'train':
        model.train_and_evaluate(
            train_input_fn, eval_input_fn, eval_steps=args.eval_steps,throttle_secs=100)
    elif args.mode == 'eval':
        res = model.evaluate(
            eval_input_fn,
            steps=args.eval_steps,
            checkpoint_path=args.ckpt_path)
        print(res)
    elif args.mode == 'predict':
        model.predict(eval_input_fn, checkpoint_path=args.ckpt_path)
    else:
        raise ValueError(
            'arg mode should be one of "train", "eval", "predict"')


if __name__ == "__main__":
    main()
