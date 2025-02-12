from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense


class AE(Model):
    def __init__(self, encoded_num, feature_num):
        super(AE, self).__init__()
        self.encoder = Sequential()
        self.decoder = Sequential()

        self.encoder.add(Dense(encoded_num, activation='relu'))
        self.decoder.add(Dense(feature_num, activation='sigmoid'))

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


class SAE(Model):
    def __init__(self, encoded_num, feature_num, units):
        super(SAE, self).__init__()
        self.encoder = Sequential()
        self.decoder = Sequential()
        # encoder
        for u in units:
            self.encoder.add(Dense(u, activation='relu'))
        self.encoder.add(Dense(encoded_num, activation='relu'))
        # decoder
        for u in reversed(units):
            self.decoder.add(Dense(u, activation='relu'))
        self.decoder.add(Dense(feature_num, activation='relu'))

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x
