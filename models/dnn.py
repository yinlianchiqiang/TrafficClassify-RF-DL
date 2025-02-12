from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import Input


class dnn(Sequential):
    def __init__(self, input_shape=40, dropout=0.05):
        super(dnn, self).__init__()
        self.params = {'input_shape': input_shape, 'dropout': dropout}

        self.add(Input(shape=(input_shape, ), name='dnn'))
        denses = [600, 500, 400, 300, 200, 100, 50]
        for dense in denses:
            self.add(Dense(dense, activation='relu'))
            self.add(Dropout(dropout))

        self.add(Dense(5, activation='softmax'))

    def get_config(self):
        config = super().get_config()
        for key in self.params.keys():
            config[key] = self.params[key]
        return config
