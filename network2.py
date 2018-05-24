#-*- coding: UTF-8 -*-

from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)


        # 输入层三个神经元作为第一层
        # 第二层定义为5个神经元
        # 第三层定义为4个神经元
        layer2 = 9
        layer3 = 4

        # 随机初始化各层权重
        self.synaptic_weights1 = 2 * random.random((3, layer2)) - 1
        self.synaptic_weights2 = 2 * random.random((layer2, layer3)) - 1
        self.synaptic_weights3 = 2 * random.random((layer3, 1)) - 1


    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))


    def __sigmoid_derivative(self, x):
        return x * (1 - x)


    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # 正向传播过程，即神经网络“思考”的过程
            activation_values2 = self.__sigmoid(dot(training_set_inputs, self.synaptic_weights1))
            activation_values3 = self.__sigmoid(dot(activation_values2, self.synaptic_weights2))
            output = self.__sigmoid(dot(activation_values3, self.synaptic_weights3))

            # 计算各层损失值
            delta4 = (training_set_outputs - output) * self.__sigmoid_derivative(output)
            delta3 = dot(self.synaptic_weights3, delta4.T) * (self.__sigmoid_derivative(activation_values3).T)
            delta2 = dot(self.synaptic_weights2, delta3) * (self.__sigmoid_derivative(activation_values2).T)

            # 计算需要调制的值
            adjustment3 = dot(activation_values3.T, delta4)
            adjustment2 = dot(activation_values2.T, delta3.T)
            adjustment1 = dot(training_set_inputs.T, delta2.T)

            # 调制权值
            self.synaptic_weights1 += adjustment1
            self.synaptic_weights2 += adjustment2
            self.synaptic_weights3 += adjustment3


    def think(self, inputs):
        activation_values2 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
        activation_values3 = self.__sigmoid(dot(activation_values2, self.synaptic_weights2))
        output = self.__sigmoid(dot(activation_values3, self.synaptic_weights3))
        return output


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("Random starting synaptic weights (layer 1): ")
    print(neural_network.synaptic_weights1)
    print("\nRandom starting synaptic weights (layer 2): ")
    print(neural_network.synaptic_weights2)
    print("\nRandom starting synaptic weights (layer 3): ")
    print(neural_network.synaptic_weights3)

    # 训练集不变
    training_set_inputs = array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1],[0, 1, 0]])
    training_set_outputs = array([[1, 1, 0, 0,0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("\nNew synaptic weights (layer 1) after training: ")
    print(neural_network.synaptic_weights1)
    print("\nNew synaptic weights (layer 2) after training: ")
    print(neural_network.synaptic_weights2)
    print("\nNew synaptic weights (layer 3) after training: ")
    print(neural_network.synaptic_weights3)

    # 新样本测试
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_network.think(array([0, 1, 1])))