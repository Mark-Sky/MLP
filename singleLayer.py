import numpy as np
from mnist import load_datasets
from utils import Sigmoid, Sigmoid_grad, accuracy, fromOneHot, softmax, drop_out_matrices

class MLP():
    def __init__(self, epoch=200,
                 init_hidden_dis='random',
                 lossfun='crossEntropyLoss',
                 lr_adjustment='fixed',
                 regulization='None',
                 is_dropout=False):

        self.epoch = epoch
        self.hidden_size = 512
        self.input_size = 28 * 28
        self.output_size = 10

        self.init_hidden_dis = init_hidden_dis # the method for the initialization of the weight
        # zero, random, xavier_guassian, xavier_uniform, He_guassian, He_uniform

        self.lossfun = lossfun # loss function
        # crossEntropyLoss, MSE

        self.lr_adjustment = lr_adjustment # param for the adjustment of the learning rate
        # fixed, section_fixed, inverse_time, exp_decay, nature_exp， cos_decay
        # CLR， SGDR
        # Warmup
        # AdaGrad， RMSProp， Adam

        self.regulization = regulization
        # L1, L2
        # Dropout

        self.is_dropout = is_dropout

        self.Wxh = np.zeros((self.hidden_size, self.input_size))
        self.Why = np.zeros((self.output_size, self.hidden_size))
        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.output_size, 1))

        # Cumulative gradient first moment for the AdaGrad/RMSProp/Adam
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

        # Cumulative gradient second moment for the Adam Optimization
        self.msWxh = np.zeros_like(self.Wxh)
        self.msWhy = np.zeros_like(self.Why)
        self.msbh = np.zeros_like(self.bh)
        self.msby = np.zeros_like(self.by)

        self.learning_rate = 0.2
        self.lr_0 = 0.2 # init learning rate

        self.lamda = 1 # for the regulization L1 and L2

        self.drop_matrices = None
        self.keep_prob = [1, 0.6, 1] # for dropout

        self.init_Hidden()

    def init_Hidden(self):
        if self.init_hidden_dis == "zero":
            pass

        if self.init_hidden_dis == "random":
            self.Wxh = np.random.randn(self.hidden_size, self.input_size) * 0.01
            self.Why = np.random.randn(self.output_size, self.hidden_size) * 0.01

        if self.init_hidden_dis == "xavier_guassian":
            self.Wxh = np.random.normal(loc=0, scale=16*(2/(self.input_size+self.hidden_size)), size=(self.hidden_size, self.input_size))
            self.Why = np.random.normal(loc=0, scale=16*(2/(self.hidden_size+self.output_size)), size=(self.output_size, self.hidden_size))

        if self.init_hidden_dis == "xavier_uniform":
            self.learning_rate = 0.5
            self.lr_0 = 0.5
            r1 = 4 * np.sqrt(6 / (self.input_size+self.hidden_size))
            self.Wxh = np.random.uniform(-r1, r1, size=(self.hidden_size, self.input_size))
            r2 = 4 * np.sqrt(6 / (self.hidden_size+self.output_size))
            self.Why = np.random.uniform(-r2, r2, size=(self.output_size, self.hidden_size))

        if self.init_hidden_dis == "He_guassian":
            self.Wxh = np.random.normal(loc=0, scale=2/(self.input_size), size=(self.hidden_size, self.input_size))
            self.Why = np.random.normal(loc=0, scale=2/(self.hidden_size), size=(self.output_size, self.hidden_size))

        if self.init_hidden_dis == "He_uniform":
            r1 = np.sqrt(6 / self.input_size)
            r2 = np.sqrt(6 / self.hidden_size)
            self.Wxh = np.random.uniform(-r1, r1, size=(self.hidden_size, self.input_size))
            self.Why = np.random.uniform(-r2, r2, size=(self.output_size, self.hidden_size))

    def init_drop_matrices(self):
        layer_dims = [self.input_size, self.hidden_size, self.output_size]
        self.drop_matrices = drop_out_matrices(layer_dims, 60000, self.keep_prob)

    def forward(self, X):
        h1 = Sigmoid(self.Wxh @ X.T + self.bh)
        y_pred = Sigmoid(self.Why @ h1 + self.by)

        return h1.T, y_pred.T

    # use different forward function for the nn with dropout
    def forward_drop(self, X):
        X_drop = np.multiply(X, self.drop_matrices[0]) / self.keep_prob[0]

        h1 = Sigmoid(self.Wxh @ X_drop.T + self.bh)
        h1_drop = np.multiply(h1, self.drop_matrices[1].T) / self.keep_prob[1]

        y_pred = Sigmoid(self.Why @ h1_drop + self.by)

        return h1.T, y_pred.T

    def backward(self, dWxh, dWhy, dbh, dby, epoch):
        if self.lr_adjustment == "AdaGrad" or self.lr_adjustment == "RMSProp":
            for param, dparam, mem in zip([self.Wxh, self.Why, self.bh, self.by],
                                          [dWxh, dWhy, dbh, dby],
                                          [self.mWxh, self.mWhy, self.mbh, self.mby]):
                if self.lr_adjustment == "AdaGrad":
                    mem += dparam * dparam
                    param += self.learning_rate * dparam / np.sqrt(mem + 1e-8)

                if self.lr_adjustment == "RMSProp":
                    rho = 0.2
                    mem += ((rho - 1) * mem + (1 - rho) * dparam * dparam)
                    param += self.learning_rate * dparam / np.sqrt(mem + 1e-8)


        elif self.lr_adjustment == "Adam" :
            for param, dparam, mem, smem in zip([self.Wxh, self.Why, self.bh, self.by],
                                          [dWxh, dWhy, dbh, dby],
                                          [self.mWxh, self.mWhy, self.mbh, self.mby],
                                          [self.msWxh, self.msWhy, self.msbh, self.msby]):

                alpha, beta1, beta2, espilon = 0.001, 0.9, 0.999, 1e-8
                mem += ((beta1 - 1) * mem + (1 - beta1) * dparam)
                smem += ((beta2 - 1) * smem + (1 - beta2) * dparam * dparam)
                mem_hat = mem / (1 - beta1 ** (epoch+1))
                smem_hat = smem / (1 - beta2 ** (epoch+1))
                param += alpha * mem_hat / (np.sqrt(smem_hat) + espilon)


        else:
            for param, dparam in zip([self.Wxh, self.Why, self.bh, self.by],
                                     [dWxh, dWhy, dbh, dby]):
                param += self.learning_rate * dparam


    def lr_adjust(self, epoch):
        if self.lr_adjustment == "AdaGrad" or self.lr_adjustment == "RMSProp":
            self.learning_rate = 0.1

        if self.lr_adjustment == "fixed":
            return

        if self.lr_adjustment == "section_fixed":
            if epoch % 10 == 1:
                if self.learning_rate > 0.1:
                    self.learning_rate -= 0.005

        if self.lr_adjustment == "inverse_time":
            self.lr_0 = 0.5
            self.learning_rate = self.lr_0 * (1 / (1 + 0.1 * epoch))

        if self.lr_adjustment == "exp_decay":
            self.lr_0 = 0.5
            self.learning_rate = self.lr_0 * 0.96 ** epoch

        if self.lr_adjustment == "nature_exp":
            self.lr_0 = 0.5
            self.learning_rate = self.lr_0 * np.exp(-0.04 * epoch)

        if self.lr_adjustment == "cos_decay":
            self.lr_0 = 0.5
            self.learning_rate = self.lr_0 * (1 + np.cos(epoch * np.pi / self.epoch)) / 2

        if self.lr_adjustment == "CLR":
            if epoch % 100 < 50:
                self.learning_rate += 0.002
            else:
                self.learning_rate -= 0.002

        if self.lr_adjustment == "SGDR":
            if epoch % 50 == 0:
                self.learning_rate = self.lr_0
            else:
                self.learning_rate = self.lr_0 * np.exp(-0.04 * (epoch % 50))

        if self.lr_adjustment == "Warmup":
            T = 10
            if epoch < T:
                self.learning_rate = epoch * self.lr_0 / T
            else:
                self.learning_rate = self.lr_0 * np.exp(-0.04 * (epoch - T))

    def loss_compute(self, X, y):

        h, y_pred = None, None
        if self.is_dropout:
            h, y_pred = self.forward_drop(X)
        else:
            h, y_pred = self.forward(X)

        dWhy = np.zeros(self.Why.shape)
        dWxh = np.zeros(self.Wxh.shape)
        dbh = np.zeros(self.bh.shape)
        dby = np.zeros(self.by.shape)
        loss = 0

        if self.lossfun == "MSE":
            loss = np.sum(np.square(y_pred - y)) / 2 / len(y)

            g = (y - y_pred) * y_pred * (1 - y_pred)

            dWhy = (g.T @ h) / X.shape[0]
            dby = -np.expand_dims(np.mean(g, axis=0), axis=1)

            e = h * (1 - h) * (np.dot(g, self.Why))
            if self.is_dropout:
                e = np.multiply(e, self.drop_matrices[1])
                e /= self.keep_prob[1]

            dWxh = np.dot(X.T, e).T / X.shape[0]
            dbh = -np.expand_dims(np.mean(e, axis=0), axis=1)

        if self.lossfun == "crossEntropyLoss":
            p = softmax(y_pred)

            loss = np.sum([-np.log(p[i][np.argmax(y[i])]) for i in range(len(p))]) / len(y)

            g = np.copy(p) - np.eye(10)[np.argmax(y, axis=1)]
            g = -g * y_pred * (1 - y_pred) # (60000, 10)

            dWhy = (g.T @ h) / X.shape[0]
            dby = -np.expand_dims(np.mean(g, axis=0), axis=1)

            e = h * (1 - h) * (np.dot(g, self.Why))
            if self.is_dropout:
                e = np.multiply(e, self.drop_matrices[1])
                e /= self.keep_prob[1]

            dWxh = np.dot(X.T, e).T / X.shape[0]
            dbh = -np.expand_dims(np.mean(e, axis=0), axis=1)

        if self.regulization == "L1":
            dWxh -= self.lamda * np.sign(self.Wxh) / len(X)
            dWhy -= self.lamda * np.sign(self.Why) / len(X)
            dbh -= self.lamda * np.sign(self.bh) / len(X)
            dby -= self.lamda * np.sign(self.by) / len(X)

        if self.regulization == "L2":
            dWxh -= 2 * self.lamda * self.Wxh / len(X)
            dWhy -= 2 * self.lamda * self.Why / len(X)
            dbh -= 2 * self.lamda * self.bh / len(X)
            dby -= 2 * self.lamda * self.by / len(X)

        return dWxh, dWhy, dbh, dby, loss

    def train(self, train_X, train_y, test_X, test_y):
        loss_list = []
        test_acc_list = []
        train_acc_list = []

        for i in range(self.epoch + 2):

            if self.is_dropout:
                self.init_drop_matrices()

            dWxh, dWhy, dbh, dby, loss = self.loss_compute(train_X, train_y)
            self.backward(dWxh, dWhy, dbh, dby, i)
            self.lr_adjust(i)
            loss_list.append(loss)

            if i % 10 == 1:
                y_pred_class = self.predict(test_X)
                test_acc = accuracy(fromOneHot(test_y), y_pred_class)
                print("epoch = ", i-1)
                print("loss = ", loss)
                print("test_acc = ", test_acc)

                test_acc_list.append(test_acc)
                train_y_pred = self.predict(train_X)
                train_acc = accuracy(fromOneHot(train_y), train_y_pred)
                print("train acc = ", train_acc)
                train_acc_list.append(train_acc)
                print('')
        return loss_list, test_acc_list, train_acc_list

    def predict(self, X):
        _, y_pred = self.forward(X)
        y_pred_class = fromOneHot(y_pred)
        return y_pred_class

if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_datasets()
    epoch = 200

    best = MLP(epoch=epoch,
               init_hidden_dis='xavier_guassian',
               lossfun='crossEntropyLoss',
               lr_adjustment='Adam')
    best.train(train_X, train_y, test_X, test_y)


    dropout = MLP(epoch=epoch, is_dropout=True)
    dropout_loss, dropout_test_acc, dropout_train_acc = dropout.train(train_X, train_y, test_X, test_y)
    print('dropout_train_acc', dropout_train_acc[-1])
    print('dropout_test_acc', dropout_test_acc[-1])


    baseline = MLP(epoch=epoch)
    base_loss, base_test_acc, base_train_acc = baseline.train(train_X, train_y, test_X, test_y)
    print('base_train_acc', base_train_acc[-1])
    print('base_test_acc', base_test_acc[-1])

    xa_guas = MLP(epoch=epoch, init_hidden_dis='xavier_guassian')
    xa_guas_loss, xa_guas_test_acc, xa_guas_train_acc = xa_guas.train(train_X, train_y, test_X, test_y)
    print('xa_guas_train_acc', xa_guas_train_acc[-1])
    print('xa_guas_test_tacc', xa_guas_test_acc[-1])

    xa_unif = MLP(epoch=epoch, init_hidden_dis='xavier_uniform')
    xa_unif_loss, xa_unif_test_acc, xa_unif_train_acc = xa_unif.train(train_X, train_y, test_X, test_y)
    print('xa_unif_train_acc', xa_unif_train_acc[-1])
    print('xa_unif_test_acc', xa_unif_test_acc[-1])


    he_guas = MLP(epoch=epoch, init_hidden_dis='He_guassian')
    he_guas_loss, he_guas_test_acc, he_guas_train_acc = he_guas.train(train_X, train_y, test_X, test_y)
    print('he_guas_train_acc', he_guas_train_acc[-1])
    print('he_guas_test_acc', he_guas_test_acc[-1])

    he_unif = MLP(epoch=epoch, init_hidden_dis='He_uniform')
    he_unif_loss, he_unif_test_acc, he_unif_train_acc = he_unif.train(train_X, train_y, test_X, test_y)
    print('he_unif_train_acc', he_unif_train_acc[-1])
    print('he_unif_test_acc', he_unif_test_acc[-1])


    zero = MLP(epoch=epoch, init_hidden_dis='zero')
    zero_loss, zero_test_acc, zero_train_acc = zero.train(train_X, train_y, test_X, test_y)
    print('zero_train_acc', zero_train_acc[-1])
    print('zero_test_acc', zero_test_acc[-1])

    mse = MLP(epoch=epoch, lossfun='MSE')
    mse_loss, mse_test_acc, mse_train_acc = mse.train(train_X, train_y, test_X, test_y)
    print('mse_train_acc', mse_train_acc[-1])
    print('mse_test_acc', mse_test_acc[-1])

    section_fixed = MLP(epoch=epoch, lr_adjustment='section_fixed')
    section_f_loss, section_f_test_acc, section_f_train_acc = section_fixed.train(train_X, train_y, test_X, test_y)
    print('section_f_train_acc', section_f_train_acc[-1])
    print('section_f_test_acc', section_f_test_acc[-1])


    inverse_time = MLP(epoch=epoch, lr_adjustment='inverse_time')
    inverse_t_loss, inverse_t_test_acc, inverse_t_train_acc = inverse_time.train(train_X, train_y, test_X, test_y)
    print('inverse_t_train_acc', inverse_t_train_acc[-1])
    print('inverse_t_test_acc', inverse_t_test_acc[-1])

    exp_decay = MLP(epoch=epoch, lr_adjustment='exp_decay')
    exp_de_loss, exp_de_test_acc, exp_de_train_acc = exp_decay.train(train_X, train_y, test_X, test_y)
    print('exp_de_train_acc', exp_de_train_acc[-1])
    print('exp_de_test_acc', exp_de_test_acc[-1])

    nature_exp = MLP(epoch=epoch, lr_adjustment='nature_exp')
    nature_exp_loss, nature_exp_test_acc, nature_exp_train_acc = nature_exp.train(train_X, train_y, test_X, test_y)
    print('nature_exp_train_acc', nature_exp_train_acc[-1])
    print('nature_exp_test_acc', nature_exp_test_acc[-1])

    cos_decay = MLP(epoch=epoch, lr_adjustment='cos_decay')
    cos_decay_loss, cos_decay_test_acc, cos_decay_train_acc = cos_decay.train(train_X, train_y, test_X, test_y)
    print('cos_decay_train_acc', cos_decay_train_acc[-1])
    print('cos_decay_test_acc', cos_decay_test_acc[-1])

    CLR = MLP(epoch=epoch, lr_adjustment='CLR')
    CLR_loss, CLR_test_acc, CLR_train_acc = CLR.train(train_X, train_y, test_X, test_y)
    print('CLR_train_acc', CLR_train_acc[-1])
    print('CLR_test_acc', CLR_test_acc[-1])

    SGDR = MLP(epoch=epoch, lr_adjustment='SGDR')
    SGDR_loss, SGDR_test_acc, SGDR_train_acc = SGDR.train(train_X, train_y, test_X, test_y)
    print('SGDR_train_acc', SGDR_train_acc[-1])
    print('SGDR_test_acc', SGDR_test_acc[-1])

    Warmup = MLP(epoch=epoch, lr_adjustment='Warmup')
    Warmup_loss, Warmup_test_acc, Warmup_train_acc = Warmup.train(train_X, train_y, test_X, test_y)
    print('Warmup_train_acc', Warmup_train_acc[-1])
    print('Warmup_test_acc', Warmup_test_acc[-1])

    AdaGrad = MLP(epoch=epoch, lr_adjustment='AdaGrad')
    AdaGrad_loss, AdaGrad_test_acc, AdaGrad_train_acc = AdaGrad.train(train_X, train_y, test_X, test_y)
    print('AdaGrad_train_acc', AdaGrad_train_acc[-1])
    print('AdaGrad_test_acc', AdaGrad_test_acc[-1])

    RMSProp = MLP(epoch=epoch, lr_adjustment='RMSProp')
    RMSProp_loss, RMSProp_test_acc, RMSProp_train_acc = RMSProp.train(train_X, train_y, test_X, test_y)
    print('RMSProp_train_acc', RMSProp_train_acc[-1])
    print('RMSProp_test_acc', RMSProp_test_acc[-1])

    Adam = MLP(epoch=epoch, lr_adjustment='Adam')
    Adam_loss, Adam_test_acc, Adam_train_acc = Adam.train(train_X, train_y, test_X, test_y)
    print('Adam_train_acc', Adam_train_acc[-1])
    print('Adam_test_acc', Adam_test_acc[-1])

    L1 = MLP(epoch=epoch, regulization='L1')
    L1_loss, L1_test_acc, L1_train_acc = L1.train(train_X, train_y, test_X, test_y)
    print('L1_train_acc', L1_train_acc[-1])
    print('L1_test_acc', L1_test_acc[-1])

    L2 = MLP(epoch=epoch, regulization='L2')
    L2_loss, L2_test_acc, L2_train_acc = L2.train(train_X, train_y, test_X, test_y)
    print('L2_train_acc', L2_train_acc[-1])
    print('L2_test_acc', L2_test_acc[-1])

    dropout = MLP(epoch=epoch, is_dropout=True)
    dropout_loss, dropout_test_acc, dropout_train_acc = dropout.train(train_X, train_y, test_X, test_y)
    print('dropout_train_acc', dropout_train_acc[-1])
    print('dropout_test_acc', dropout_test_acc[-1])


