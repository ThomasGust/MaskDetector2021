from tensorflow.keras.layers import Conv2D, Dropout, Activation, MaxPooling2D, Layer, BatchNormalization


class ConvBlock(Layer):
    def __init__(self, convs, dropout=[True, 0.2], pool=[(2, 2), None], first=False, input_shape=(None, 128, 128, 3)):
        super(ConvBlock, self).__init__()
        self.convs = []
        if type(convs) is not list:
            print(
                f"Parameter 'convs' of class ConvBlock was not the required type of list, found type was {type(convs)}")
        if type(dropout) is not list:
            print(
                f"Parameter 'dropout' of class ConvBlock was not the required type of list, found type was {type(dropout)}")
        if type(pool) is not list:
            print(f"Parameter 'pool' of class ConvBlock was not the required type of list, found type was {type(pool)}")

        for index in range(len(convs)):
            if convs[index][2] is None:
                convs[index].pop(2)

        for c in range(len(convs)):
            #print(len(convs[c]))
            if len(convs[c]) > 2:
                if first is True and input_shape is not None and len(self.convs) < 1:
                    self.convs.append(Conv2D(filters=convs[c][0], kernel_size=convs[c][1],
                                             strides=convs[c][2], input_shape=input_shape))
                else:
                    self.convs.append(Conv2D(filters=convs[c][0], kernel_size=convs[c][1],
                                             strides=convs[c][2]))
            else:
                if first is True and input_shape is not None and len(self.convs) < 1:
                    self.convs.append(Conv2D(filters=convs[c][0], kernel_size=convs[c][1],
                                             input_shape=input_shape))
                else:
                    self.convs.append(Conv2D(filters=convs[c][0], kernel_size=convs[c][1],
                                             input_shape=input_shape))
        self.convs.append(BatchNormalization(axis=1))
        self.has_dropout = dropout[0]
        self.drop_rate = dropout[1]

        if self.has_dropout is True:
            self.drop = Dropout(self.drop_rate)
        self.pool = MaxPooling2D(pool[0], pool[1])
        self.a = Activation('relu')

    def call(self, inputs, training=False, *args, **kwargs):
        x = inputs
        for i, conv in enumerate(self.convs):
            cl = self.convs[i]
            x = cl(x)
        x = self.pool(x)
        x = self.a(x)

        if self.has_dropout is True:
            return self.drop(x)
        else:
            return x
