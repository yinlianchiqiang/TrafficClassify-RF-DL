from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input,  Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import Conv1D, MaxPooling1D, Conv1DTranspose, UpSampling1D


class CAE_1d_1500(Model):
    def __init__(self, filter_num=512, dropout=0.15):
        super(CAE_1d_1500, self).__init__()
        self.encoder = Sequential()
        self.decoder = Sequential()

        #  encoder
        self.encoder.add(Input(shape=(1500, 1), name='CAE_1500'))
        self.encoder.add(Conv1D(filter_num, kernel_size=2, activation='relu'))  # (None, 1499, 16)
        self.encoder.add(Dropout(dropout))
        self.encoder.add(Conv1D(filter_num, kernel_size=4, activation='relu'))  # (None, 1496, 16)
        self.encoder.add(Dropout(dropout))
        self.encoder.add(MaxPooling1D(pool_size=2))  # (None, 748, 16)
        self.encoder.add(Flatten())  # (None, 11968)
        self.encoder.add(Dense(512, activation='relu'))

        #  decoder
        self.decoder.add(Dense(11968, activation='sigmoid'))
        self.decoder.add(Reshape((748, 16)))
        self.decoder.add(UpSampling1D(2))
        self.decoder.add(Conv1DTranspose(filter_num, 4, activation='sigmoid'))
        self.decoder.add(Conv1DTranspose(filter_num, 2, activation='sigmoid'))
        self.decoder.add(Conv1D(1, 1, activation='sigmoid'))

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


class CAE_2d_2500(Model):
    def __init__(self, filter_num=512, dropout=0.15):
        super(CAE_2d_2500, self).__init__()
        self.encoder = Sequential()
        self.decoder = Sequential()
        #  encoder
        self.encoder.add(Input(shape=(50, 50, 1), name='CAE_2500'))
        self.encoder.add(Conv2D(filter_num, kernel_size=(1, 3), activation='relu'))  # (None, 50, 48, 16)
        self.encoder.add(Dropout(dropout))
        self.encoder.add(Conv2D(filter_num, kernel_size=(4, 4), activation='relu', padding='same'))  # (None, 50, 48, 16)
        self.encoder.add(Dropout(dropout))
        self.encoder.add(MaxPooling2D((2, 2)))  # (None, 25, 24, 16) 
        self.encoder.add(Flatten())  # (None, 9600)
        self.encoder.add(Dense(512, activation='relu'))

        #  decoder
        self.decoder.add(Dense(9600, activation='sigmoid'))
        self.decoder.add(Reshape((25, 24, 16) ))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2DTranspose(filter_num, (4, 4), activation='sigmoid', padding='same'))
        self.decoder.add(Conv2DTranspose(filter_num, (1, 3), activation='sigmoid'))
        self.decoder.add(Conv2D(1, (1, 1), activation='sigmoid'))

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
