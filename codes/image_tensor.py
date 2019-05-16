
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
import time
# from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def initialize_weights_biases(layers):
    weights_list = []
    biases_list = []
    layers_length = len(layers)
    for l in range(0,layers_length-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        weights_list.append(W)
        biases_list.append(b)
    return weights_list, biases_list


def xavier_init(size):
    input_dimsion = size[0]
    output_dimsion = size[1]
    xavier_stddev = np.sqrt(2/(input_dimsion + output_dimsion))
    return tf.Variable(tf.truncated_normal([input_dimsion, output_dimsion], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)

def NNwork(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.sin(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y


###############################################################################

class DeepHPM:
    def __init__(self, t, x, u,
                       u_layers, pde_layers,
                       layers,
                       lb_idn, ub_idn,
                       ):

        # Domain Boundary
        self.lb_idn = lb_idn
        self.ub_idn = ub_idn

        # self.lb_sol = lb_sol
        # self.ub_sol = ub_sol

        # Init for Identification
        self.idn_init(t, x, u, u_layers, pde_layers)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))




        init = tf.global_variables_initializer()
        self.sess.run(init)

    ###########################################################################
    ############################# Identifier ##################################
    ###########################################################################

    def idn_init(self, t, x, u, u_layers, pde_layers):
        # Training Data for Identification
        self.t = t
        self.x = x
        self.u = u
        # to add a one
        self.one = np.full_like(self.u, 1.0)


        # Layers for Identification
        self.u_layers = u_layers
        self.pde_layers = pde_layers

        # Initialize NNs for Identification
        self.u_weights, self.u_biases = initialize_weights_biases(u_layers)
        self.pde_weights, self.pde_biases = initialize_weights_biases(pde_layers)

        # tf placeholders for Identification
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.terms_tf = tf.placeholder(tf.float32, shape=[None, pde_layers[0]])
        # add a one
        self.one_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # tf graphs for Identification
        self.idn_u_pred = self.idn_net_u(self.t_tf, self.x_tf)
        self.pde_pred = self.net_pde(self.terms_tf)
        self.idn_f_pred = self.idn_net_f(self.t_tf, self.x_tf, self.u_tf)

        # loss for Identification
        self.idn_u_loss = tf.reduce_sum(tf.square(self.idn_u_pred - self.u_tf))
        self.idn_f_loss = tf.reduce_sum(tf.square(self.idn_f_pred))

        # my version
        # shape_of_u = self.idn_u_pred.get_shape()
        # shape_of_u = shape_of_u.as_list()
        # one = tf.ones(shape_of_u, dtype=tf.float32)
        # tf.to_float(one)
        # self.idn_u_loss = tf.reduce_sum(tf.multiply(tf.nn.tanh(tf.square(self.idn_u_pred - self.u_tf)),tf.square(self.idn_u_pred - self.u_tf)))
        # self.idn_f_loss = tf.reduce_sum(tf.subtract(one,tf.multiply(tf.nn.tanh(tf.square(self.idn_u_pred - self.u_tf))),tf.square(self.idn_f_pred)))
        # self.idn_f_loss = tf.reduce_sum(tf.multiply(tf.subtract(self.one_tf,tf.nn.tanh(tf.square(self.idn_u_pred - self.u_tf))),tf.square(self.idn_f_pred)))
        # self.idn_f_loss = tf.reduce_sum(tf.multiply(tf.nn.tanh(tf.square(self.idn_u_pred - self.u_tf)),tf.square(self.idn_f_pred)))
        self.idn_u_f_loss = self.idn_u_loss + self.idn_f_loss



        # optimizer for combiantion
        self.idn_u_f_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_u_f_loss,
                               var_list = self.u_weights + self.u_biases + self.pde_weights + self.pde_biases,
                               method = 'L-BFGS-B',
                               options = {'maxiter': 100000,
                                          'maxfun': 100000,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})

        # Optimizer for Identification
        self.idn_u_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_u_loss,
                               var_list = self.u_weights + self.u_biases,
                               method = 'L-BFGS-B',
                               options = {'maxiter': 10000,
                                          'maxfun': 10000,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})

        self.idn_f_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_f_loss,
                               var_list = self.pde_weights + self.pde_biases,
                               method = 'L-BFGS-B',
                               options = {'maxiter': 10000,
                                          'maxfun': 10000,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})

        self.idn_u_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_u_train_op_Adam = self.idn_u_optimizer_Adam.minimize(self.idn_u_loss,
                                   var_list = self.u_weights + self.u_biases)

        self.idn_f_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_f_train_op_Adam = self.idn_f_optimizer_Adam.minimize(self.idn_f_loss,
                                   var_list = self.pde_weights + self.pde_biases)
        # for combination
        self.idn_u_f_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_u_f_train_op_Adam = self.idn_f_optimizer_Adam.minimize(self.idn_u_f_loss,
                                   var_list = self.pde_weights + self.pde_biases + self.u_weights + self.u_biases)

    def idn_net_u(self, t, x):
        X = tf.concat([t,x],1)
        H = 2.0*(X - self.lb_idn)/(self.ub_idn - self.lb_idn) - 1.0
        u = NNwork(H, self.u_weights, self.u_biases)
        return u

    def net_pde(self, terms):
        pde = NNwork(terms, self.pde_weights, self.pde_biases)
        return pde

    def idn_net_f(self, t, x, u0):
        u = self.idn_net_u(t, x)


        u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]

        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        terms = tf.concat([u,u_xx,u0],1)

        f = u_tt - self.net_pde(terms)
        return f

    def idn_u_f_train(self,N_iter):
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x, self.u_tf: self.u,self.one_tf:self.one}

        start_time = time.time()
        for it in range(N_iter):

            self.sess.run(self.idn_u_f_train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.idn_u_f_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.idn_u_f_optimizer.minimize(self.sess,
                                      feed_dict = tf_dict,
                                      fetches = [self.idn_u_f_loss],
                                      loss_callback = self.callback)

    def idn_u_train(self, N_iter):
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x, self.u_tf: self.u}

        start_time = time.time()
        for it in range(N_iter):

            self.sess.run(self.idn_u_train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.idn_u_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.idn_u_optimizer.minimize(self.sess,
                                      feed_dict = tf_dict,
                                      fetches = [self.idn_u_loss],
                                      loss_callback = self.callback)

    def idn_f_train(self, N_iter):
        # tf_dict = {self.t_tf: self.t, self.x_tf: self.x}
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x,self.u_tf: self.u, self.one_tf:self.one}

        start_time = time.time()
        for it in range(N_iter):

            self.sess.run(self.idn_f_train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.idn_f_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.idn_f_optimizer.minimize(self.sess,
                                      feed_dict = tf_dict,
                                      fetches = [self.idn_f_loss],
                                      loss_callback = self.callback)



    def idn_predict(self, t_star, x_star):

        tf_dict = {self.t_tf: t_star, self.x_tf: x_star}

        u_star = self.sess.run(self.idn_u_pred, tf_dict)
        f_star = self.sess.run(self.idn_f_pred, tf_dict)

        return u_star, f_star

    def predict_pde(self, terms_star):

        tf_dict = {self.terms_tf: terms_star}

        pde_star = self.sess.run(self.pde_pred, tf_dict)

        return pde_star
    #

    def callback(self, loss):
        print('Loss: %e' % (loss))





###############################################################################
################################ Main Function ################################
###############################################################################

if __name__ == "__main__":

    # Doman bounds
    lb_idn = np.array([0.0, 0.0])
    ub_idn = np.array([255.0, 255.0])

    ### Load Data ###

    # data_idn = scipy.io.loadmat('../Data/burgers.mat')
    im = plt.imread("Lenna.jpg")
    im_0 = im[:, :, 0]
    data_idn = im_0
    imsize = im_0.shape
    # t_idn = data_idn['t'].flatten()[:,None]
    # x_idn = data_idn['x'].flatten()[:,None]
    # Exact_idn = np.real(data_idn['usol'])
    #
    # T_idn, X_idn = np.meshgrid(t_idn,x_idn)
    x_list = []
    y_list = []
    u_list = []
    for i in range(imsize[0]):
        for j in range(imsize[0]):
            x_list.append(np.array([i]))
            y_list.append(np.array([j]))
            u_list.append(np.array([im_0[i][j]]))


    x_array = np.array(x_list)
    y_array = np.array(y_list)

    u_array = np.array(u_list)
    ### Training Data ###

    # For identification
    N_train = 10000

    idx = imsize[0]*imsize[0]
    # idx = np.random.choice(t_idn_star.shape[0], N_train, replace=False)
    # t_train = t_idn_star[idx,:]
    # x_train = x_idn_star[idx,:]
    # u_train = u_idn_star[idx,:]
    x_train = []
    y_train = []
    u_train = []
    for i in range(idx):
        x_train.append(np.array(x_array[i]))
        y_train.append(np.array(y_array[i]))
        u_train.append(np.array(u_array[i]))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    u_train = np.array(u_train)
    # print(im_0)
    # print(x_train)
    # print(y_train)
    # print(u_train)


    # # print(im_0)
    # print("u train")
    # print(u_train)
    noise = 0.05
    u_train = u_train + noise * np.std(u_train)* np.random.randn(u_train.shape[0], u_train.shape[1])
    #
    # # set up layer
    # # Layers
    u_layers = [2, 50, 50, 50, 50, 1]
    pde_layers = [3, 100, 100, 1]

    layers = [2, 50, 50, 50, 50, 1]

    model = DeepHPM(y_train, x_train, u_train,
                    u_layers, pde_layers,
                    layers,
                    lb_idn, ub_idn,
                    )

    # Train the identifier
    model.idn_u_train(N_iter=0)

    model.idn_f_train(N_iter=0)

    # model.idn_u_f_train(N_iter=0)

    u_pred_identifier, f_pred_identifier = model.idn_predict(y_array, x_array)

    error_u_identifier = np.linalg.norm(u_array-u_pred_identifier,2)/np.linalg.norm(u_array,2)
    print('Error u: %e' % (error_u_identifier))
    print(u_array)
    print(u_pred_identifier)
    im_learn = []
    im_noise = []
    for i in range(imsize[0]):
        newline = []
        newlinenoise = []
        for j in range(imsize[0]):
            newline.append(u_pred_identifier[i*imsize[0]+j][0])
            newlinenoise.append(u_train[i*imsize[0]+j][0])
        im_learn.append(newline)
        # print(newlinenoise)
        im_noise.append(newlinenoise)

    im_noise = np.array(im_noise)
    im_learn = np.array(im_learn)


    print("origianl")
    print(im_0)
    print("im_noise")
    print(im_noise)
    print("im_learn")
    print(im_learn)
    plt.subplot(1,2,1)
    plt.imshow(im_noise)
    plt.subplot(1,2,2)
    plt.imshow(im_learn)
    plt.show()
    # #


# # original
# stren=200
# sigma=0.18
# alpha=0.3
# im = plt.imread("Lenna.jpg")
# im_0=im[:,:,0]
# print(im_0[0][1])
# imsize=im_0.shape
# im_noise=stren*np.random.normal(loc=0, scale=sigma, size=imsize)+im_0
#
# Delta=np.zeros(imsize)
# for i in range(1,imsize[0]-1):
#     for j in range(1,imsize[1]-1):
#         Delta[i, j] = im_noise[i + 1, j] + im_noise[i - 1, j] + im_noise[i, j + 1] + im_noise[i, j - 1] - 4 * im_noise[i, j]
#
# im_recover=im_noise+alpha*Delta
#
# plt.subplot(1,2,1)
# plt.imshow(im_noise)
# plt.subplot(1,2,2)
# plt.imshow(im_recover)
# plt.show()
