import numpy as np
import scipy.special as sp
import pandas as pd
class net:
    '''
    BPnerualnetwork
    '''
    def __init__(self, sample_data, output_data, hidden_num1,error=0.001, rate=0.8):
        sample_num, input_num = np.shape(sample_data)
        self.sample_data = sample_data
        self.output_data = output_data
        self.sample_num = sample_num
        self.input_num = input_num
        self.hidden_num1 = hidden_num1
        self.rate = rate
        self.error = error

        self.max_data = max(self.output_data)
        self.min_data = min(self.output_data)
        self.output_data = (self.output_data-self.min_data)/(self.max_data-self.min_data)


        '''
        初始化矩阵
        '''
        self.w1 = np.random.rand(hidden_num1, input_num) * 2 - 1
        self.w2 = np.random.rand(hidden_num2, hidden_num1) * 2 - 1
        self.w3 = np.random.rand(1, hidden_num2) * 2 - 1

        self.b1 = np.random.rand(hidden_num1, 1) * 2 - 1
        self.b2 = np.random.rand(hidden_num2, 1) * 2 - 1
        self.b3 = np.random.rand(1,1) * 2 - 1

        self.train()

    def train(self):
        for i in range(self.sample_num):
            input_data = self.sample_data[i]
            lable  = self.output_data[i]
            input_data = input_data.reshape(self.input_num,1)
            lable = lable.reshape(1,1)

            input_hidden1 = np.dot(self.w1, input_data)+self.b1
            output_hidden1 = sp.expit(input_hidden1)
            output_hidden1 = output_hidden1.reshape(hidden_num1,1)
            input_hidden2 = np.dot(self.w2, output_hidden1)+self.b2
            output_hidden2 = sp.expit(input_hidden2)
            output_hidden2 = output_hidden2.reshape(hidden_num2, 1)
            input_out = np.dot(self.w3, output_hidden2)+self.b3
            final = sp.expit(input_out)

            E3 = final * (1 - final) * (lable - final)

            while (abs(E3) >= self.error):
                E2 = output_hidden2*(1-output_hidden2)*np.dot(self.w3.T,E3)
                E1 = output_hidden1*(1-output_hidden1)*np.dot(self.w2.T,E2)

                self.w1 += self.rate * E1 * input_data.T
                self.w2 += self.rate * E2 * output_hidden1.T
                self.w3 += self.rate * E3 * output_hidden2.T

                self.b1 += self.rate * E1
                self.b2 += self.rate * E2
                self.b3 += self.rate * E3

                input_hidden1 = np.dot(self.w1, input_data)+self.b1
                output_hidden1 = sp.expit(input_hidden1)
                output_hidden1 = output_hidden1.reshape(hidden_num1, 1)
                input_hidden2 = np.dot(self.w2, output_hidden1)+self.b2
                output_hidden2 = sp.expit(input_hidden2)
                output_hidden2 = output_hidden2.reshape(hidden_num2, 1)
                input_out = np.dot(self.w3, output_hidden2)+self.b3
                final = sp.expit(input_out)
                E3 = final * (1 - final) * (lable - final)

        print('successfully finish train!')


    def privite(self, privite_data):
        privite_data = privite_data.reshape(self.input_num, 1)
        input_hidden1 = np.dot(self.w1, privite_data)+self.b1
        output_hidden1 = sp.expit(input_hidden1)
        output_hidden1 = output_hidden1.reshape(hidden_num1, 1)
        input_hidden2 = np.dot(self.w2, output_hidden1)+self.b2
        output_hidden2 = sp.expit(input_hidden2)
        output_hidden2 = output_hidden2.reshape(hidden_num2, 1)
        input_out = np.dot(self.w3, output_hidden2)+self.b3
        final = sp.expit(input_out)
        return (final*(self.max_data-self.min_data)+self.min_data)

inputfile = 'E:\project\datatime.xlsx'
inputfile1 = 'E:\project\d.xlsx'
data = pd.read_excel(inputfile)
data1 = pd.read_excel(inputfile1)
sample_data =np.array(data)
output_data = np.array(data1)

hidden_num1 = 2
hidden_num2 = 1
a = net(sample_data,output_data,hidden_num1,hidden_num2)
privite_data = np.array([600,34.6,4.2])
y = a.privite(privite_data)
print("预测的蝗虫密度为：",y[0][0])
