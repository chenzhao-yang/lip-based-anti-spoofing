# -*- coding: utf-8 -*-

import tensorflow as tf

def read_images(images_raw,channel):
    """ Read raw images to tensor.
        For example T raw image will be read to Tensor of
        shape (T, h, w, channel)

    Args:
        images_raw: 1-d `string` Tensor. Each element is an encoded jpeg image.
        size: Tuple (h, w).  The image will be resized to such size.
        channel: Int. 1 will output grayscale images, 3 outputs RGB
                 images.

    Returns: 4-D `float32` Tensor. The decoded images.
    """
    i = tf.constant(0)
    image_length = tf.shape(images_raw)[0]
    images = tf.TensorArray(dtype=tf.float32, size=image_length)

    condition = lambda i, images: tf.less(i, image_length)

    def loop_body(i, images):
        """ The loop body of reading images.
        """
        
        image= tf.image.decode_jpeg(images_raw[i], channels=channel)
        image = tf.image.resize_images(image,size=[40,60],method=tf.image.ResizeMethod.BILINEAR)
        image=tf.cast(image,tf.float32)
        images = images.write(i, image)
      
        return tf.add(i, 1) , images

    i, images = tf.while_loop(
        condition,
        loop_body,
        [i, images],
        back_prop=False,
        # parallel_iterations=VIDEO_LENGTH
    )
    x = images.stack()  # T x H x W x C
    
    return x

def optical_parse_function(example_proto):
    features = {"video": tf.VarLenFeature(tf.string),
                'motion':tf.VarLenFeature(tf.string),
                "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    video = parsed_features['video']

    video = tf.sparse_tensor_to_dense(video, default_value='')
     
    video = read_images(video, 3)
    video/=255

    motion = parsed_features['motion']
    motion = tf.sparse_tensor_to_dense(motion, default_value='')
    motion = tf.decode_raw(motion, tf.float32)
    motion= tf.reshape(motion, (-1,40,60,2))

    inputs = {
                'video': video[:16, ...],
                'motion':motion[:15, ...]
                }
    targets = {
                'label': parsed_features["label"]}
     return (inputs,targets)

def tfrecord_input_fn(file_name_pattern,
                           mode=tf.estimator.ModeKeys.EVAL,
                           num_epochs=1,
                           batch_size=32,
                           num_threads=4):
    """TODO: Docstring for grid_tfrecord_input_fn.

    Args:
        file_name_pattern: tfrecord filenames

    Kwargs:
        mode: train or others. Local shuffle will be performed if train.
        num_epochs: repeat data num_epochs times.
        batch_size: batch_size
        num_threads:

    Returns: TODO

    """
    file_names = tf.matching_files(file_name_pattern)
    dataset = tf.data.TFRecordDataset(filenames=file_names,num_parallel_reads=4)
    dataset = dataset.map(optical_parse_function, num_parallel_calls=num_threads)

    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100 * batch_size + 1)

    dataset = dataset.repeat(num_epochs)

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=({
            'video':[16 ,40, 64, 3],
            'motion':[15, 40, 64, 2]
         },
         {
            'label': []
         })
        )
    dataset = dataset.prefetch(buffer_size=10)
    return dataset
