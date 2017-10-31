# coding: UTF-8

# -----------------------------------------------------------------------------
# Copyright (C.) 2017 Jastec Corp., No.3 Development Div. All rights reserved.
#
#  Program Name: train_mnist.py
#  [mnistデータを使ってCNNを訓練(学習)する]
#  Contributors: Naoshi WATANUKI  -- Initial API and implementation. --
#  Data Written: 2017.9.12(Tue.)
#  Update Written: ____.__.__
#  SourceTree committed: 2017.9.12(Tue.)
# -----------------------------------------------------------------------------

import sys
import argparse
import tensorflow as tf

# [MNIST Data 読込み] ---------------*
#  Training data : 55,000 rec(points)
#  Test data     : 10,000 rec(points)
# -----------------------------------*
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

# [Download data_dir] -------------------------------------------*
# mnist.pyでMNIST Datasetsをダウンロードする
# mnist.pyの場所は,TensorFlowをインストールした環境に依存する
# mnist.pyの詳細はTensorFlowのチュートリアルサイトを参照すること。
#
#  Download folder: ./MNIST_data/
#  one_hot: One_hotエンコーディングをTrueにする
# ---------------------------------------------------------------*
def practice():
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  
  # Interactive Session()を開始する
  sess = tf.InteractiveSession()

  # [Input placeholders] -------------------------------------------------------* 
  #  ２次元(2-dim(28px x 28px))の入力画像をflatten処理により、
  #  １次元(1-dim(784px))に変換して入力する。 従って、784の入力ニューロンとなる。
  #  また、y_は、推定値であり、softmax関数により0-9の10個の出力クラスを持つ
  # ----------------------------------------------------------------------------*
  with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  # Output: flatten処理により、1D(784px)データに変換された入力画像を、reshape(28px x 28px)に戻す
  with tf.name_scope('Input_Reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('Input', image_shaped_input, 10)

  # weightを定義して初期化する
  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  # biasを定義して初期化する
  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  # TensorBoard 表示用のサマリー編集を行う(-> tutorialからコピペ)
  def variable_summaries(var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  # ---------------------------------------------------------------------------*
  # Neural Networkレイヤー(nn_layer)を構成する
  # 入力値に対してweightを掛け、さらにbiasを加えた総和を求め、ReLU関数で
  # 非線形処理を行う。その後、name_scope処理を行い、集計結果グラフを見易くする
  # ---------------------------------------------------------------------------*
  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('Pre_Activations', preactivate)
      activations = act(preactivate, name='Activation')
      tf.summary.histogram('Activations', activations)
      return activations

  # TensorFlow tutorialを参考にして、中間層(隠れ層)Neuronを700に変更!!
  middle_layer = nn_layer(x, 784, 700, 'Layer1')

  with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('Keep_Probability', keep_prob)
    dropped = tf.nn.dropout(middle_layer, keep_prob)

  y = nn_layer(dropped, 700, 10, 'Middle_Layer', act=tf.identity)

  # [交差エントロピー誤差検出] --------------------------------*
  #  nn_layerの入力に対するweight,biasの総和(出力結果)に対して、
  #  tf.nn.softmax_cross_entropy_with_logitsを使って誤差算出し、
  #  その後、平均を求める
  # -----------------------------------------------------------*
  with tf.name_scope('Cross_Entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('Total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('Cross_Entropy', cross_entropy)

  with tf.name_scope('Practice'):
    practice_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

  with tf.name_scope('Accuracy'):
    with tf.name_scope('Correct_Prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('Accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('Accuracy', accuracy)

  # [全結合 -> TensorBoard] ----------------------------------------------*
  #  nn_layerの計算結果(集計結果)を全結合し、下記のディレクトリに書き出す
  #  --logdir=/tmp/tensorflow/logs/train_mnist
  # ----------------------------------------------------------------------*
  merged = tf.summary.merge_all()
  practice_writer = tf.summary.FileWriter(FLAGS.log_dir + '/practice', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # ---------------------------------------------------------------------------------------*
  # modelの学習を行い、集計結果を書き出す
  # 10回学習するごとに、test-setの正確性を検証し、集計結果を書き出す
  # practice_stepメソッドでtraining dataの訓練（学習）を行い、訓練結果(サマリー)を書き出す
  # ---------------------------------------------------------------------------------------*
  def feed_dict(practice):
    if practice:
      xs, ys = mnist.train.next_batch(100)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  print('     ')
  print('[画像認識　訓練開始 Copyright(C) 2017 株式会社ジャステック 製造3部 . Allright Reserved. ]')
  print('     ')
  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      acc *= 100
      print('   訓練の正確さ Step %03d ->  %03.2f' % (i, acc) + ' %')
    else:
      if i % 100 == 99:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run(
                            [merged, practice_step],
                            feed_dict=feed_dict(True),
                            options=run_options,
                            run_metadata=run_metadata
                            )
        practice_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        practice_writer.add_summary(summary, i)
        print('     ')
        print(' Mini Batch 処理 : ', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, practice_step], feed_dict=feed_dict(True))
        practice_writer.add_summary(summary, i)
  print('     ')
  print('--- End of Deep Learning (CNN), See ya ! ---')
  practice_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  practice()


if __name__ == '__main__':
 # ------------------*
 # argparse defines. *
 # ------------------*
  print(' ')
  parser = argparse.ArgumentParser(
          prog='train_mnist.py',
          usage='Training for CNN by MNIST(classics).',
          epilog='--- End of HELP, see ya! ---',
          description='[MNISTを使用して画像認識回路(CNN)の訓練を行います]')
  parser.add_argument(
          '--max_steps', 
          type=int, 
          default=1700,
          help='訓練の実行回数(Default=1700回)')
  parser.add_argument(
          '--learning_rate', 
          type=float, 
          default=0.001,
          help='学習率(Default=0.001)')
  parser.add_argument(
          '--dropout', 
          type=float, 
          default=0.9,
          help='過剰学習制御(Default=0.9)')
  parser.add_argument(
          '--data_dir',
          type=str,
          default='/tmp/tensorflow/mnist/input_data',
          help='入力データ(Default:/tmp/tensorflow/mnist/input_data)')
  parser.add_argument(
          '--log_dir',
          type=str,
          default='/tmp/tensorflow/logs/train_mnist',
          help='訓練ログ(Default:/tmp/tensorflow/logs/train_mnist)')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
