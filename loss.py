import math
import abc
import numpy as np

class LossFunction(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def initialize_f_0(self, data):
        """Init F_0 """

    @abc.abstractmethod
    def calculate_residual(self, data, iter):
        """Calc negative gradient"""

    @abc.abstractmethod
    def update_f_m(self, data, trees, iter, learning_rate, logger):
        """Calc F_m """

    @abc.abstractmethod
    def update_leaf_values(self, targets, y):
        """Update leaf node's val"""

    @abc.abstractmethod
    def get_train_loss(self, y, f, iter, logger):
        """Calc Train loss"""


class SquaresError(LossFunction):

    def initialize_f_0(self, data):
        data['f_0'] = data['label'].mean()
        return data['label'].mean()

    def calculate_residual(self, data, iter):
        res_name = 'res_' + str(iter)
        f_prev_name = 'f_' + str(iter - 1)
        data[res_name] = data['label'] - data[f_prev_name]

    def update_f_m(self, data, trees, iter, learning_rate):
        f_prev_name = 'f_' + str(iter - 1)
        f_m_name = 'f_' + str(iter)
        data[f_m_name] = data[f_prev_name]
        for leaf_node in trees[iter].leaf_nodes:
            data.loc[leaf_node.remain_indexs, f_m_name] += learning_rate * leaf_node.predict_value
            
        #self.get_train_loss(data['label'], data[f_m_name], iter)

    def update_leaf_values(self, targets, y):
        numerator = targets.sum()
        if numerator == 0:
            return 0.0    
        return targets.mean()

    def get_train_loss(self, y, f, iter):
        loss = ((y - f) ** 2).mean()
        return loss


class BinomialDeviance(LossFunction):

    def initialize_f_0(self, data):
        #Todo: should chage the lable each time
        pos = sum(data['label'] == 1)
        neg = data.shape[0] - pos
        f_0 = np.log(pos / neg)
        data['f_0'] = f_0
        return f_0

    def calculate_residual(self, data, iter):
        # calculate negative gradient
        res_name = 'res_' + str(iter)
        f_prev_name = 'f_' + str(iter - 1)
        data[res_name] = data['label'] - (1 / (1 + np.exp(-data[f_prev_name])))
        #data[res_name] = data['label'] - 1 / (1 + data[f_prev_name].apply(lambda x: np.exp(-x)))

    def update_f_m(self, data, trees, iter, learning_rate):
        f_prev_name = 'f_' + str(iter - 1)
        f_m_name = 'f_' + str(iter)
        data[f_m_name] = data[f_prev_name]
        for leaf_node in trees[iter].leaf_nodes:
            data.loc[leaf_node.remain_indexs, f_m_name] += learning_rate * leaf_node.predict_value
        
        return self.get_train_loss(data['label'], data[f_m_name], iter)

    def update_leaf_values(self, targets, y):
        numerator = targets.sum()
        if numerator == 0:
            return 0.0
        denominator = ((y - targets) * (1 - y + targets)).sum()
        if abs(denominator) < 1e-150:
            return 0.0
        else:
            return numerator / denominator

    def get_train_loss(self, y, f, iter):
        loss = -2.0 * ((y * f) - f.apply(lambda x: np.exp(1+x))).mean()
        return loss


class MultinomialDeviance:

    def init_classes(self, classes):
        self.classes = classes

    @abc.abstractmethod
    def initialize_f_0(self, data, class_name):
        label_name = 'label_' + class_name
        f_name = 'f_' + class_name + '_0'
        class_counts = data[label_name].sum()
        f_0 = class_counts / len(data)
        data[f_name] = f_0
        return f_0

    def calculate_residual(self, data, iter):
        # calculate negative gradient
        data['sum_exp'] = data.apply(lambda x:
                                     sum([np.exp(x['f_' + i + '_' + str(iter - 1)]) for i in self.classes]),
                                     axis=1)
        for class_name in self.classes:
            label_name = 'label_' + class_name
            res_name = 'res_' + class_name + '_' + str(iter)
            f_prev_name = 'f_' + class_name + '_' + str(iter - 1)
            data[res_name] = data[label_name] - np.e ** data[f_prev_name] / data['sum_exp']

    def update_f_m(self, data, trees, iter, class_name, learning_rate):
        f_prev_name = 'f_' + class_name + '_' + str(iter - 1)
        f_m_name = 'f_' + class_name + '_' + str(iter)
        data[f_m_name] = data[f_prev_name]
        for leaf_node in trees[iter][class_name].leaf_nodes:
            data.loc[leaf_node.remain_indexs, f_m_name] += learning_rate * leaf_node.predict_value
        
        #self.get_train_loss(data['label'], data[f_m_name], iter)

    def update_leaf_values(self, targets, y):
        numerator = targets.sum()
        if numerator == 0:
            return 0.0
        numerator *= (self.classes.size - 1) / self.classes.size
        denominator = ((y - targets) * (1 - y + targets)).sum()
        if abs(denominator) < 1e-150:
            return 0.0
        else:
            return numerator / denominator

    def get_train_loss(self, y, f, iter):
        loss = -2.0 * ((y * f) - f.apply(lambda x: np.exp(1+x))).mean()
        return loss

