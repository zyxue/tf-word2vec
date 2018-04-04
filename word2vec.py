from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile
import logging

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

import pandas as pd

logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s|%(levelname)s|%(message)s')


def get_args():
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir', type=str, default=os.path.join(current_path, 'log'),
        help='The log directory for TensorBoard summaries.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help=''
    )
    parser.add_argument(
        '--embedding_size', type=int, default=128,
        help='Dimension of the embedding vector.'
    )
    parser.add_argument(
        '--skip_window', type=int, default=1,
        help=('How many words to consider left and right. '
              'aka context size/window size')
    )
    parser.add_argument(
        '--num_steps', type=int, default=100001,
        help=('Number of training steps')
    )
    return parser.parse_args()


# Step 1: Download the data.
# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes, url='http://mattmahoney.net/dc/'):
    """Download a file if not present, and make sure it's the right size."""
    local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(
            url + filename, local_filename)
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename +
                        '. Can you get to it with a browser?')
    return local_filename


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:    # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# TODO: remove dependency on global variable
DATA_INDEX = 0


def generate_batch(batch_size, num_skips, skip_window):
    global DATA_INDEX
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1    # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if DATA_INDEX + span > len(data):
        DATA_INDEX = 0
    buffer.extend(data[DATA_INDEX:DATA_INDEX + span])
    DATA_INDEX += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if DATA_INDEX == len(data):
            buffer.extend(data[0:span])
            DATA_INDEX = span
        else:
            buffer.append(data[DATA_INDEX])
            DATA_INDEX += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    DATA_INDEX = (DATA_INDEX + len(data) - span) % len(data)
    return batch, labels


class SkipgramModel(object):
    def __init__(self, config):
        self.config = config

    def define_inputs(self):
        batch_size = self.config['batch_size']
        valid_examples = self.config['valid_examples']
        # Input data.
        with tf.name_scope('inputs'):
            self.train_xs = tf.placeholder(tf.int32, shape=[batch_size])
            self.train_ys = tf.placeholder(tf.int32, shape=[batch_size, 1])
            self.valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    def define_computation(self):
        vocabulary_size = self.config['vocabulary_size']
        embedding_size = self.config['embedding_size']

        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            # the embedding matrix
            embeddings = tf.Variable(tf.random_uniform(
                [vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, self.train_xs)

        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                    tf.truncated_normal(
                            [vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        self.embeddings = embeddings
        self.embed = embed
        self.nce_weights = nce_weights
        self.nce_biases = nce_biases

    def define_loss(self):
        num_sampled = self.config['num_sampled']
        vocabulary_size = self.config['vocabulary_size']
        # Compute the average NCE loss for the batch. tf.nce_loss automatically
        # draws a new sample of the negative labels each time we evaluate the
        # loss. Explanation of the meaning of NCE loss:
        # http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weights,
                    biases=self.nce_biases,
                    labels=self.train_ys,
                    inputs=self.embed,
                    num_sampled=num_sampled,
                    num_classes=vocabulary_size))

    def define_summary(self):
        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', self.loss)

        # Merge all summaries.
        self.summary = tf.summary.merge_all()

    def define_optimizer(self):
        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer
            self.optimizer = optimizer(1.0).minimize(self.loss)

    def define_similarity(self):
        # Compute the cosine similarity between minibatch examples and all
        # embeddings.
        norm = tf.sqrt(tf.reduce_sum(
            tf.square(self.embeddings), 1, keep_dims=True))

        normalized_embeddings = self.embeddings / norm

        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, self.valid_dataset)

        similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

        self.normalized_embeddings = normalized_embeddings
        self.similarity = similarity

    def build_graph(self):
        self.define_inputs()
        self.define_computation()
        self.define_loss()
        self.define_summary()
        self.define_optimizer()
        self.define_similarity()

    def train(self):
        num_steps = self.config['num_steps']
        log_dir = self.config['log_dir']
        valid_examples = self.config['valid_examples']

        # Create a saver.
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Open a writer to write summaries.
            writer = tf.summary.FileWriter(self.config['log_dir'], sess.graph)

            # We must initialize all variables before we use them.
            init = tf.global_variables_initializer()
            init.run()

            average_loss = 0
            for step in xrange(num_steps):

                batch_xs, batch_ys = generate_batch(
                    batch_size=self.config['batch_size'],
                    num_skips=self.config['num_skips'],
                    skip_window=self.config['skip_window'],
                )
                feed_dict = {
                    self.train_xs: batch_xs,
                    self.train_ys: batch_ys
                }

                # Define metadata variable.
                run_metadata = tf.RunMetadata()

                # We perform one update step by evaluating the optimizer op
                # (including it in the list of returned values for sess.run()
                # Also, evaluate the merged op to get all summaries from the
                # returned "summary" variable. Feed metadata variable to sess
                # for visualizing the graph in TensorBoard.
                _, summary, loss_val = sess.run(
                    [self.optimizer, self.summary, self.loss],
                    feed_dict=feed_dict,
                    run_metadata=run_metadata)
                average_loss += loss_val

                writer.add_summary(summary, step)
                if step == (num_steps - 1):
                    writer.add_run_metadata(run_metadata, 'step%d' % step)

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last
                    # 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every
                # 500 steps)
                if step % 10000 == 0:
                    sim = self.similarity.eval()
                    for i in xrange(valid_size):
                        valid_word = reverse_dictionary[valid_examples[i]]
                        top_k = 8    # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in xrange(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        logging.info(log_str)

            self.final_embeddings = self.normalized_embeddings.eval()

            # Write corresponding labels for the embeddings.
            with open(log_dir + '/metadata.tsv', 'w') as f:
                for i in xrange(vocabulary_size):
                    f.write(reverse_dictionary[i] + '\n')

            # Save the model for checkpoints.
            saver.save(sess, os.path.join(log_dir, 'model.ckpt'))

            # Create a configuration for visualizing embeddings with the labels
            # in TensorBoard.
            config = projector.ProjectorConfig()
            embedding_conf = config.embeddings.add()
            embedding_conf.tensor_name = self.embeddings.name
            embedding_conf.metadata_path = os.path.join(
                self.config['log_dir'], 'metadata.tsv')
            projector.visualize_embeddings(writer, config)

        writer.close()


if __name__ == "__main__":
    args = get_args()
    # Create the directory for TensorBoard variables if there is not.
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    filename = maybe_download('text8.zip', 31344016)

    vocabulary = read_data(filename)
    logging.info('Data size: {0}'.format(len(vocabulary)))

    # Step 2: Build the dictionary and replace rare words with UNK token.
    vocabulary_size = 50000

    # Filling 4 global variables:
    # data - list of codes (integers from 0 to vocabulary_size-1).
    #     This is the original text but words are replaced by their codes
    # count - map of words(strings) to count of occurrences
    # dictionary - map of words(strings) to their codes(integers)
    # reverse_dictionary - maps codes(integers) to words(strings)
    data, count, dictionary, reverse_dictionary = build_dataset(
        vocabulary, vocabulary_size)
    del vocabulary    # Hint to reduce memory.
    logging.info('Most common words (+UNK): {0}'.format(count[:5]))
    logging.info('Sample data, {0}, {1}'.format(
        data[:10], [reverse_dictionary[i] for i in data[:10]]))

    valid_size = 16
    valid_window = 100
    # valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_examples = np.arange(16)

    model = SkipgramModel(config=dict(
        log_dir=args.log_dir,

        batch_size=args.batch_size,
        embedding_size=args.embedding_size,
        skip_window=args.skip_window,
        num_steps=args.num_steps,

        # How many times to reuse an input to generate a label.
        num_skips=2,
        # Number of negative examples to sample.
        num_sampled=64,

        vocabulary_size=vocabulary_size,
        dictionary=dictionary,
        reverse_dictionary=reverse_dictionary,

        # We pick a random validation set to sample nearest neighbors. Here we
        # limit the validation samples to the words that have a low numeric ID,
        # which by construction are also the most frequent. These 3 variables
        # are used only for displaying model accuracy, they don't affect
        # calculation.

        # Random set of words to evaluate similarity on.
        valid_size=valid_size,
        # Only pick dev samples in the head of the distribution.
        valid_window=valid_window,
        valid_examples=valid_examples,
    ))

    model.build_graph()
    model.train()

    # save emebeddings
    df_out = pd.DataFrame(model.final_embeddings)
    labels_all = [reverse_dictionary[i] for i in xrange(
        model.final_embeddings.shape[0])]
    df_out['label'] = labels_all
    out = 'embeddings_{0}D_{1}W.csv'.format(
        args.embedding_size, args.skip_window)
    df_out.to_csv(out, index=False)
