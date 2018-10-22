import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # Get tensors to return
    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    def conv_1x1(x, num_classes):
        kernel_size = 1
        #stride = 1
        return tf.layers.conv2d(x, num_classes, kernel_size, strides=(1, 1), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    conv7 = conv_1x1_lay7 = conv_1x1(vgg_layer7_out, num_classes)
    deconv7 = tf.layers.conv2d_transpose(conv7, num_classes, 4,2, 'SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    
    conv4 = conv_1x1_lay4 = conv_1x1(vgg_layer4_out, num_classes)
    add_deconv7_conv4 = tf.add(deconv7, conv4)  # first skip layer
    deconv_add_deconv7_conv4 = tf.layers.conv2d_transpose(add_deconv7_conv4, num_classes, 4, 2, 'SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    
    conv3 = output = conv_1x1(vgg_layer3_out, num_classes)
    add_conv3_deconv_add_deconv7_conv4 = tf.add(deconv_add_deconv7_conv4, conv3)  # second skip layer
    last_layer_output = tf.layers.conv2d_transpose(add_conv3_deconv_add_deconv7_conv4, num_classes, 16, 8, 'same',kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))  # scale up by x8 to get original image size
    return last_layer_output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    logits, optimizer, cross_entropy_loss = optimize(last_layer_output, correct_label, learning_rate, num_classes)
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Reshape predictions and labels
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    # Define loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Define train_op to minimise loss
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    for epoch in range(epochs):
        for batch, (image, label) in enumerate(get_batches_fn(batch_size)):
            feed_dict = {input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 1e-4}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            print('Epoch ', epoch, ' Batch ', batch, ' Loss ', loss, flush=True)
    pass
tests.test_train_nn(train_nn)


def run():
    # Set parameters
    num_classes = 2 #road or no road
    image_shape = (160, 576)
    epochs = 1#20
    batch_size = 4
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    
    helper.maybe_download_pretrained_vgg(data_dir) # Download pretrained vgg model if not already there

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        learning_rate = tf.placeholder(tf.float32)
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        #Load Pretrained VGG Model into TensorFlow and return: Tuple of Tensors from VGG model 
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        last_layer_output= layers(layer3_out, layer4_out, layer7_out, num_classes)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        logits, train_op, cross_entropy_loss = optimize(last_layer_output, correct_label, learning_rate, num_classes)

        
        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)


        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image: image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
