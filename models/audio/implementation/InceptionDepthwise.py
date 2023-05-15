from keras.models import Model
from keras.layers import Dense, Conv2D, AveragePooling2D, BatchNormalization, DepthwiseConv2D
from keras.layers.merge import concatenate


InceptionDepthwise(Model)
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', strides=(1, 1), activation='relu')

        self.avg_pool1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', strides=(1, 1), activation='relu')

        self.conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', strides=(1, 1), activation='relu')
        self.batch_norm1 = BatchNormalization()
        self.depth_conv1 = DepthwiseConv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')

        self.conv4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', strides=(1, 1), activation='relu')
        self.batch_norm2 = BatchNormalization()
        self.depth_conv2 = DepthwiseConv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')
        self.batch_norm3 = BatchNormalization()
        self.depth_conv3 = DepthwiseConv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')

        self.conv5 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', strides=(1, 1), activation='relu')
        self.batch_norm4 = BatchNormalization()
        self.depth_conv4 = DepthwiseConv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')
        self.batch_norm5 = BatchNormalization()
        self.depth_conv5 = DepthwiseConv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')
        self.batch_norm6 = BatchNormalization()
        self.depth_conv6 = DepthwiseConv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')


     def call(self, inputs):
        out1 = self.conv_1(inputs)
        
        out2 = self.avg_pool1(inputs)
        out2 = self.conv2(out2)
        
        out3 = self.conv3(inputs)
        out3 = self.batch_norm1(out3)
        out3 = self.depth_conv1(out3)
        
        out4 = self.conv4(inputs)
        out4 = self.batch_norm2(out4)
        out4 = self.depth_conv2(out4)
        out4 = self.batch_norm3(out4)
        out4 = self.depth_conv3(out4)
        
        out5 = self.conv5(inputs)
        out5 = self.batch_norm4(out5)
        out5 = self.depth_conv4(out5)
        out5 = self.batch_norm5(out5)
        out5 = self.depth_conv5(out5)
        out5 = self.batch_norm6(out5)
        out5 = self.depth_conv6(out5)
        
        final_layer = concatenate([out1, out2, out3, out4, out5], axis=-1)
        return final_layer
    