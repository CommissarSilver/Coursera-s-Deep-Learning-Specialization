import tensorflow as tf

# input_shape = ()
layers = []
conv_layers_num = input("Let's get building your CNN\n How many Convolutional Layers does it have?")

for i in range(int(conv_layers_num)):
    print('***parameters for conv layer number {0}***'.format(i + 1))
    units_num = input('How many units in this layer: ')
    kernel_size = input("what's the kernel Size: (enter as comma separated)").split(',')
    stride_size = input("What's the stride's size: ")
    batch_norm_bool = input('Do you want to use batch normalization: (y or n)')
    dropout_bool = input('Should we do dropout: (y or n)')

    if dropout_bool == 'y':
        dropout_rate = input("What's the dropout rate: ")
    pooling_bool = input("Do you want a pooling layer: (y or n)")

    if pooling_bool == 'y':
        pool_type = input("Average or Max pooling: (avg or max)")
        pool_kernel_size = input("What's kernel's size: ")

    layers.append(
        tf.keras.layers.Conv2D(int(units_num), kernel_size=tuple([int(x) for x in kernel_size]), strides=int(stride_size),
                               activation='relu'))

    if dropout_bool == 'y':
        layers.append(tf.keras.layers.Dropout(int(dropout_rate)))

    if pooling_bool == 'y':
        if pool_type == 'max':
            layers.append(tf.keras.layers.MaxPooling2D(pool_size=pool_kernel_size))
        else:
            layers.append(tf.keras.layers.AveragePooling2D(pool_size=pool_kernel_size))

layers.append(tf.keras.layers.Flatten())
dense_layers_num = input(
    "Now that we're done with the conv layers let's get to dense layers\n How many dense layer do you want?")

for i in range(int(dense_layers_num)):
    print('***parameters for dense layer number {0}***'.format(i + 1))
    units_num = input('How many units in this layer: ')

    if i+1 != int(dense_layers_num):
        dropout_bool = input('Should we do dropout: (y or n)')
        if dropout_bool == 'y':
            dropout_rate = input("What's the dropout rate: ")
        layers.append(tf.keras.layers.Dense(units=int(units_num), activation='relu'))
        if dropout_bool == 'y':
            layers.append(tf.keras.layers.Dropout(int(dropout_rate)))
    else:
        layers.append(tf.keras.layers.Dense(units=int(units_num), activation='softmax'))

r = tf.keras.models.Sequential(layers)

r.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

r.summary()
