from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, GlobalMaxPooling1D
from keras import Input, Model

class CNN_2d(Sequential):
    def __init__(self, input_shape=32, filter_num = 16, dropout=0.05, is_dnn=True):
        super(CNN_2d, self).__init__()
        self.params = {'input_shape': input_shape, 'dropout': dropout}

        self.add(Conv2D(filter_num, kernel_size=(1, 2), activation='relu', input_shape=(input_shape, input_shape, 1)))
        self.add(Dropout(dropout))
        self.add(Conv2D(filter_num, kernel_size=(4, 4), activation='relu'))
        self.add(Dropout(dropout))
        self.add(MaxPooling2D(pool_size=2))
        
        self.add(Flatten())

        print(is_dnn)
        if is_dnn:
            denses = [600, 500, 400, 300, 200, 100, 50]
            for dense in denses:
                self.add(Dense(dense, activation='relu'))
                self.add(Dropout(dropout))
        #self.add(Dense(5, activation='softmax')) 5 是分類的類別數
        self.add(Dense(5, activation='softmax'))

    def get_config(self):
        config = super().get_config()
        for key in self.params.keys():
            config[key] = self.params[key]
        return config
    
class CNN_1d(Sequential):
    def __init__(self, input_shape=40, filter_num = 50, dropout=0.05, is_dnn=True):
        super(CNN_1d, self).__init__()
        self.params = { 'input_shape': input_shape, 'dropout': dropout}

        self.add(Conv1D(filter_num, kernel_size=2, activation='relu', input_shape=(input_shape, 1)))
        self.add(Dropout(dropout))
        self.add(Conv1D(filter_num, kernel_size=4, activation='relu'))
        self.add(Dropout(dropout))
        self.add(MaxPooling1D(pool_size=2))
        self.add(Flatten())
        
        if is_dnn:
            denses = [600, 500, 400, 300, 200, 100, 50]
            for dense in denses:
                self.add(Dense(dense, activation = 'relu'))
                self.add(Dropout(dropout))
            
        self.add(Dense(5, activation='softmax'))

    def get_config(self):
        config = super().get_config()
        for key in self.params.keys():
            config[key] = self.params[key]
        return config


def CNN_2d_graphlet(input_shape=50, filter_num = 16, dropout=0.05, is_dnn=True):
    input_cnn = Input(shape=(input_shape, input_shape, 1), name='cnn')
    input_graphlet = Input(shape=(59,), name='graphlet')

    x = Conv2D(filter_num, kernel_size=(2, 2), activation='relu')(input_cnn)
    x = Dropout(dropout)(x)
    x = Conv2D(filter_num, kernel_size=(4, 4), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D(pool_size=2)(x)
    cnn_output = Flatten()(x)
    y = Concatenate(axis=1)([input_graphlet, cnn_output])

    if is_dnn:
        denses = [600, 400, 200, 100, 50]
        for dense in denses:
            y = Dense(dense, activation='relu')(y)
            y = Dropout(dropout)(y)

    y = Dense(5, activation='softmax')(y)
    
    return Model(inputs=[input_graphlet, input_cnn], outputs=y)