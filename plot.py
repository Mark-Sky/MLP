from singleLayer import MLP
from mnist import load_datasets
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams['axes.facecolor']='snow'
from matplotlib.ticker import MaxNLocator

def plot(compare):
    train_X, train_y, test_X, test_y = load_datasets()
    epoch = 200
    acc_index = [i * 10 for i in range(int(epoch / 10 + 1))]
    loss_index = list(range(epoch + 2))
    fig, ax = plt.subplots(1, 3)
    baseline = MLP(epoch=epoch)
    base_loss, base_test_acc, base_train_acc = baseline.train(train_X, train_y, test_X, test_y)
    ax[0].plot(loss_index, base_loss, label='baseline')
    ax[1].plot(acc_index, base_test_acc, label='baseline')
    ax[2].plot(acc_index, base_train_acc, label='baseline')

    if compare == 'parameter_init':
        print('对比不同参数初始化')
        xa_guas = MLP(epoch=epoch, init_hidden_dis='xavier_guassian')
        xa_guas_loss, xa_guas_test_tacc, xa_guas_train_tacc = xa_guas.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, xa_guas_loss, label='xavier+guassian')
        ax[1].plot(acc_index, xa_guas_test_tacc, label='xavier+guassian')
        ax[2].plot(acc_index, xa_guas_train_tacc, label='xavier+guassian')

        xa_unif = MLP(epoch=epoch, init_hidden_dis='xavier_uniform')
        xa_unif_loss, xa_unif_test_acc, xa_unif_train_acc = xa_unif.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, xa_unif_loss, label='xavier+uniform')
        ax[1].plot(acc_index, xa_unif_test_acc, label='xavier+uniform')
        ax[2].plot(acc_index, xa_unif_train_acc, label='xavier+uniform')

        he_guas = MLP(epoch=epoch, init_hidden_dis='He_guassian')
        he_guas_loss, he_guas_test_acc, he_guas_train_acc = he_guas.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, he_guas_loss, label='he+guassian')
        ax[1].plot(acc_index, he_guas_test_acc, label='he+guassian')
        ax[2].plot(acc_index, he_guas_train_acc, label='he+guassian')

        he_unif = MLP(epoch=epoch, init_hidden_dis='He_uniform')
        he_unif_loss, he_unif_test_acc, he_unif_train_acc = he_unif.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, he_unif_loss, label='he+uniform')
        ax[1].plot(acc_index, he_unif_test_acc, label='he+uniform')
        ax[2].plot(acc_index, he_unif_train_acc, label='he+uniform')

        zero = MLP(epoch=epoch, init_hidden_dis='zero')
        zero_loss, zero_test_acc, zero_train_acc = zero.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, zero_loss, label='zero')
        ax[1].plot(acc_index, zero_test_acc, label='zero')
        ax[2].plot(acc_index, zero_train_acc, label='zero')

        plt.suptitle('不同的参数初始化')

    if compare == 'loss_function':
        print('对比不同损失函数')
        mse = MLP(epoch=epoch, lossfun='MSE')
        mse_loss, mse_test_acc, mse_train_acc = mse.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, mse_loss, label='MSE')
        ax[1].plot(acc_index, mse_test_acc, label='MSE')
        ax[2].plot(acc_index, mse_train_acc, label='MSE')
        plt.suptitle("不同损失函数")

    if compare == 'learning_rate_adjust1':
        print('对比固定衰减的学习率')
        section_fixed = MLP(epoch=epoch, lr_adjustment='section_fixed')
        section_f_loss, section_f_test_acc, section_f_train_acc = section_fixed.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, section_f_loss, label='subsection')
        ax[1].plot(acc_index, section_f_test_acc, label='subsection')
        ax[2].plot(acc_index, section_f_train_acc, label='subsection')

        inverse_time = MLP(epoch=epoch, lr_adjustment='inverse_time')
        inverse_t_loss, inverse_t_test_acc, inverse_t_train_acc = inverse_time.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, inverse_t_loss, label='inverse time')
        ax[1].plot(acc_index, inverse_t_test_acc, label='inverse_time')
        ax[2].plot(acc_index, inverse_t_train_acc, label='inverse_time')

        exp_decay = MLP(epoch=epoch, lr_adjustment='exp_decay')
        exp_de_loss, exp_de_test_acc, exp_de_train_acc = exp_decay.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, exp_de_loss, label='exp_decay')
        ax[1].plot(acc_index, exp_de_test_acc, label='exp_decay')
        ax[2].plot(acc_index, exp_de_train_acc, label='exp_decay')

        nature_exp = MLP(epoch=epoch, lr_adjustment='nature_exp')
        nature_exp_loss, nature_exp_test_acc, nature_exp_train_acc = nature_exp.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, nature_exp_loss, label='nature_exp')
        ax[1].plot(acc_index, nature_exp_test_acc, label='nature_exp')
        ax[2].plot(acc_index, nature_exp_train_acc, label='nature_exp')

        cos_decay = MLP(epoch=epoch, lr_adjustment='cos_decay')
        cos_decay_loss, cos_decay_test_acc, cos_decay_train_acc = cos_decay.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, cos_decay_loss, label='cos_decay')
        ax[1].plot(acc_index, cos_decay_test_acc, label='cos_decay')
        ax[2].plot(acc_index, cos_decay_train_acc, label='cos_decay')

        plt.suptitle('固定衰减')

    if compare == 'learning_rate_adjust2':
        print('对比CLR和SGDR和学习率预热')
        CLR = MLP(epoch=epoch, lr_adjustment='CLR')
        CLR_loss, CLR_test_acc, CLR_train_acc = CLR.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, CLR_loss, label='CLR')
        ax[1].plot(acc_index, CLR_test_acc, label='CLR')
        ax[2].plot(acc_index, CLR_train_acc, label='CLR')

        SGDR = MLP(epoch=epoch, lr_adjustment='SGDR')
        SGDR_loss, SGDR_test_acc, SGDR_train_acc = SGDR.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, SGDR_loss, label='SGDR')
        ax[1].plot(acc_index, SGDR_test_acc, label='SGDR')
        ax[2].plot(acc_index, SGDR_train_acc, label='SGDR')

        Warmup = MLP(epoch=epoch, lr_adjustment='Warmup')
        Warmup_loss, Warmup_test_acc, Warmup_train_acc = Warmup.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, Warmup_loss, label='Warmup')
        ax[1].plot(acc_index, Warmup_test_acc, label='Warmup')
        ax[2].plot(acc_index, Warmup_train_acc, label='Warmup')
        plt.suptitle("学习率预热和周期变化")

    if compare == 'learning_rate_adjust3':
        print('对比自适应学习率')
        AdaGrad = MLP(epoch=epoch, lr_adjustment='AdaGrad')
        AdaGrad_loss, AdaGrad_test_acc, AdaGrad_train_acc = AdaGrad.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, AdaGrad_loss, label='AdaGrad')
        ax[1].plot(acc_index, AdaGrad_test_acc, label='AdaGrad')
        ax[2].plot(acc_index, AdaGrad_train_acc, label='AdaGrad')

        RMSProp = MLP(epoch=epoch, lr_adjustment='RMSProp')
        RMSProp_loss, RMSProp_test_acc, RMSProp_train_acc = RMSProp.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, RMSProp_loss, label='RMSProp')
        ax[1].plot(acc_index, RMSProp_test_acc, label='RMSProp')
        ax[2].plot(acc_index, RMSProp_train_acc, label='RMSProp')

        Adam = MLP(epoch=epoch, lr_adjustment='Adam')
        Adam_loss, Adam_test_acc, Adam_train_acc = Adam.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, Adam_loss, label='Adam')
        ax[1].plot(acc_index, Adam_test_acc, label='Adam')
        ax[2].plot(acc_index, Adam_train_acc, label='Adam')

        plt.suptitle("自适应学习率")

    if compare == 'regulization':
        print('对比正则化')
        L1 = MLP(epoch=epoch, regulization='L1')
        L1_loss, L1_test_acc, L1_train_acc = L1.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, L1_loss, label='L1')
        ax[1].plot(acc_index, L1_test_acc, label='L1')
        ax[2].plot(acc_index, L1_train_acc, label='L1')

        L2 = MLP(epoch=epoch, regulization='L2')
        L2_loss, L2_test_acc, L2_train_acc = L2.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, L2_loss, label='L2')
        ax[1].plot(acc_index, L2_test_acc, label='L2')
        ax[2].plot(acc_index, L2_train_acc, label='L2')

        dropout = MLP(epoch=epoch, is_dropout=True)
        dropout_loss, dropout_test_acc, dropout_train_acc = dropout.train(train_X, train_y, test_X, test_y)
        ax[0].plot(loss_index, dropout_loss, label='dropout')
        ax[1].plot(acc_index, dropout_test_acc, label='dropout')
        ax[2].plot(acc_index, dropout_train_acc, label='dropout')
        plt.suptitle('正则化')

    ax[0].legend()
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')

    ax[1].legend()
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('test accuracy')

    ax[2].legend()
    ax[2].set_xlabel('epoch')
    ax[2].set_ylabel('train accuracy')

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()



if __name__ == '__main__':

    plot(compare='parameter_init')
    plot(compare='loss_function')
    plot(compare='learning_rate_adjust1')
    plot(compare='learning_rate_adjust2')
    plot(compare='learning_rate_adjust3')
    plot(compare='regulization')


