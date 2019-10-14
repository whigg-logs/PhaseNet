
import tensorflow as tf 
import models.PickNetBase as model 

class PickNet():
    def __init__(self, batch_size=32, length_data=3000, n_channel=3, is_training=True, model_name="wavenet"):
        n_dim = 128
        self.graph = tf.Graph() 
        self.model_name = model_name 
        with self.graph.as_default():
            self.is_training = is_training
            self.input_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, length_data, n_channel])
            self.label_p = tf.placeholder(dtype=tf.int32, shape=[batch_size, length_data])
            self.label_s = tf.placeholder(dtype=tf.int32, shape=[batch_size, length_data])
            self.label_pt = tf.placeholder(dtype=tf.float32, shape=[batch_size, length_data, 1])
            self.label_st = tf.placeholder(dtype=tf.float32, shape=[batch_size, length_data, 1]) 
            self.weight_p = tf.placeholder(dtype=tf.float32, shape=[batch_size, length_data]) 
            self.weight_s = tf.placeholder(dtype=tf.float32, shape=[batch_size, length_data])
            self.weight_pt = tf.placeholder(dtype=tf.float32, shape=[batch_size, length_data])
            self.weight_st = tf.placeholder(dtype=tf.float32, shape=[batch_size, length_data])

            if model_name == "wavenet":
                net = model.wavenet(self.input_data, is_training=is_training) 
            elif model_name == "unet": 
                net = model.unet(self.input_data, is_training=is_training) 
            elif model_name == "brnn":
                net = model.brnn(self.input_data, is_training=is_training) 
            elif model_name == "inception":
                net = model.inception(self.input_data, is_training=is_training) 
            else:
                raise "Model name error"
            with tf.variable_scope('logit_p'):
                self.logit_p = tf.layers.conv1d(net, 2, 3, activation=None, padding="same") 
            with tf.variable_scope('logit_s'):
                self.logit_s = tf.layers.conv1d(net, 2, 3, activation=None, padding="same") 
            with tf.variable_scope('time_p'):
                self.times_p = tf.layers.conv1d(net, 1, 3, activation=None, padding="same") 
            with tf.variable_scope('time_s'):
                self.times_s = tf.layers.conv1d(net, 1, 3, activation=None, padding="same") 
            loss_p = tf.contrib.seq2seq.sequence_loss(self.logit_p, self.label_p, self.weight_p)
            loss_s = tf.contrib.seq2seq.sequence_loss(self.logit_s, self.label_s, self.weight_s)
            loss_tp = tf.reduce_mean(tf.reduce_sum(tf.squeeze((self.label_pt - self.times_p)**2) * self.weight_pt, axis=1))
            loss_ts = tf.reduce_mean(tf.reduce_sum(tf.squeeze((self.label_st - self.times_s)**2) * self.weight_st, axis=1))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            reg_loss = tf.losses.get_regularization_loss()
            with tf.control_dependencies(update_ops):
                self.loss = loss_p * 1 + loss_s * 1 + loss_tp * 1 + loss_ts * 1 + 1e-6*reg_loss

            # optimizer
            optimizer = tf.train.AdamOptimizer()    
            self.optimize = optimizer.minimize(self.loss)

            self.logit_loss = loss_p + loss_s
            self.times_loss = loss_tp + loss_ts
            self.nan = tf.is_nan(self.loss) 
            self.inf = tf.is_inf(self.loss)
            self.all_var = tf.trainable_variables() 
            self.init = tf.global_variables_initializer() 
            self.saver = tf.train.Saver()
        
            for itr in self.all_var:
                print(itr.name, itr.get_shape())
        self.summary = tf.summary.FileWriter("logdir", graph=self.graph)


    def init_sess(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        files = tf.train.latest_checkpoint(self.model_name)
        init_num = 0
        if files != None:
            self.saver.restore(self.sess, files)
            init_num = int(files.split('-')[-1])
        self.init_num = init_num 
        return init_num 
    def train(self, a1, a2, a3, a4, a5, a6, a7, a8, a9):
        _, ls, lgl, ltl, _ = self.sess.run([self.optimize, self.loss, self.logit_loss, self.times_loss, self.nan], feed_dict={
            self.input_data:a1, 
            self.label_p:a2,
            self.label_s:a3,
            self.label_pt:a4,
            self.label_st:a5,
            self.weight_p:a6,
            self.weight_s:a7,
            self.weight_pt:a8,
            self.weight_st:a9
        })
        return ls, lgl, ltl 
    def valid(self, x):
        prob_p = self.sess.run(tf.nn.softmax(self.logit_p), feed_dict={self.input_data:x}) 
        prob_s = self.sess.run(tf.nn.softmax(self.logit_s), feed_dict={self.input_data:x}) 
        reg_p = self.sess.run(self.times_p, feed_dict={self.input_data:x}) 
        reg_s = self.sess.run(self.times_s, feed_dict={self.input_data:x}) 
        return prob_p, prob_s, reg_p, reg_s 
    def save(self, itr=0):
        self.saver.save(self.sess, self.model_name+"/picknet", itr) 
from utils import * 
import time
import datetime 
import argparse 
def main(args):
    model = PickNet(args.batchsize, is_training=True, length_data=args.nsamples//args.resample, n_channel=args.channel, model_name=args.modelname) 
    data_tool = DataTool(args.batchsize, n_samples=args.nsamples, n_rsp=args.resample, is_3c=(args.channel==3))
    init = model.init_sess() 
    loss_file = open("wavenet-3c.txt", "a")
    avg_loss = np.ones([50]) * 7
    for itr in range(init, 1000000):
        begin = datetime.datetime.now()
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = data_tool.batch_data() 
        loss, lpl, ltl = model.train(a1, a2, a3, a4, a5, a6, a7, a8, a9) 
        loss_file.write("%d, %f\n"%(itr, loss))
        loss_file.flush() 
        avg_loss[itr%50] = loss
        end = datetime.datetime.now()
        print("Epoch:{}  Iter:{}  Time:{}  Loss:{:.3f}  AVGLoss:{:.3f}  LogitLoss:{:.3f}  TimeLoss:{:.3f}".format(data_tool.epoch, itr, str(end-begin), loss, np.mean(avg_loss), lpl, ltl))
        if itr % 50 == 0:
            print(itr, data_tool.epoch, loss)
            model.save(itr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("-b", "--batchsize", type=int, help="BatchSize", default=32) 
    parser.add_argument("-m", "--modelname", type=str, choices=["wavenet", "brnn", "inception", "unet"], 
            help="Model name: wavenet, unet, inception, brnn", default="inception") 
    parser.add_argument("-rs", "--resample", type=int, 
            help="Resample waveform", default=10) 
    parser.add_argument("-c", "--channel", type=int, 
            help="n components", default=3)       
    parser.add_argument("-n", "--nsamples", type=int, 
            help="n samples", default=30000)      
    args = parser.parse_args()
    main(args)
