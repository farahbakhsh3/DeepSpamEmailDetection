import pandas as pd
data = pd.read_csv('./train/train.csv')
x = data['Text'].to_numpy()
y = data['Class'].to_numpy()
print(x.shape)
x[8712] = ' '

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, LSTM, Input, Dropout
from tensorflow.keras.layers import Conv1D, AveragePooling1D, MaxPooling1D
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam

maxlen = 1000
max_words = 20000
embedding_dim = 50

tokenizer = text.Tokenizer(
    filters='0123456789!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    num_words=max_words)
tokenizer.fit_on_texts(x)
x_train = tokenizer.texts_to_sequences(x)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
print(x_train.shape)
y_train = y

inp = Input(shape=(maxlen,))
x = Embedding(max_words, 
                    embedding_dim, 
                    input_length=maxlen) (inp)
x = Conv1D(32,3,strides=1,
                 padding='same', activation='relu') (x)
x = MaxPooling1D(3) (x)
x = LSTM(32) (x)
x = Flatten() (x)
x = Dense(128, activation='relu') (x)
x = Dropout(0.5) (x)
out = Dense(1, activation='sigmoid') (x)

model = Model(inp, out)
model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001, beta_1=0.5),
              loss='binary_crossentropy',
              metrics=[AUC(), 'accuracy'])

history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=128,
                    validation_split=0.2)

