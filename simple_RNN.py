import tensorflow as tf
import numpy as np




max_feeature=10000
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.imdb.load_data(num_words=max_feeature)
# print the shape of data
print("shape of each split data",x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# mapping word index back to words
befor_d = tf.keras.datasets.imdb.get_word_index() # it is in form of dictionary , we revers it because we convert = index : word
after_d = {value:key for key , value in befor_d.items()}

# optional
# decoded_review = ' '.join(after_d.get(i - 3, '?') for i in sample)
# decoded_review # it is first review on x_train dataset

# from tensorflow.keras.preprocessing import sequence # for padding

# all data convert into same size using pad_sequense
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=500)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=500)

# create simple Rnn
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(10000,output_dim= 128,input_length=500,))
model.add(tf.keras.layers.SimpleRNN(128,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

# early stopiing
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)

# train the model with early stoping
history = model.fit(x_train,y_train,epochs=24,batch_size=100,callbacks=early,validation_split=0.2,validation_data=(x_test,y_test))


# Save model - modern recommended way
model.save("imdb_with_simple_RNN.keras")  # Just specify .keras extension



