import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.cross_validation import train_test_split

# Dataset path
DATASET_PATH = "/Users/bvarella/Google Drive/Algotrading NEW/Algotrading/ABEV3201701.csv"

# Load dataset
stocks_data = pandas.read_csv(DATASET_PATH)
used_features = 'fechamento_anterior'

# Splitting data
train_x, test_x, train_y, test_y = train_test_split(stocks_data[used_features], stocks_data[used_features], train_size = 0.7)

# 2-class logistic regression
model = Sequential()
model.add(Dense(1, activation = 'sigmoid', input_shape = test_x.shape[1]))
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(test_x, test_y, nb_epoch = 10, validation_data = (train_x, train_y))
score = model.evaluate(test_x, test_y, verbose = 0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
