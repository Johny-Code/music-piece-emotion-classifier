from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Add
from keras.layers.merge import concatenate
from implementation.InceptionDepthwise import InceptionDepthwise


InceptionResnet(Model):
    def __init__(self):
        super().__init__()
        self.max_pool1 = MaxPooling2D(pool_size=(2,2), stides=None, padding="same")
        self.conv1 = Conv2D(filters=160, kernel_size=(1, 1), padding='same', strides=(1, 1), activation='relu')
        self.batch_norm1 = BatchNormalization()
        
        self.inception1 = InceptionDepthwise()
        # self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu') #not described properly
        self.max_pool2 = MaxPooling2D(pool_size=(2,2), strides=None, padding="same")
        self.inception2 = InceptionDepthwise()
        
        self.add1 = Add()
        
        
    def call(self, inputs):
        skip_out = self.max_pool1(inputs)
        skip_out = self.conv1(inputs)
        skip_out = self.batch_norm1(inputs)
        
        out1 = self.inception1(inputs)
        out1 = self.max_pool2(inputs)
        out1 = self.inception2(inputs)
        
        final_layer = self.add1([skip_out, out1])
        return final_layer