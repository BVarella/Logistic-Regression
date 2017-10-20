import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split

# Dataset path
DATASET_PATH = "/Users/bvarella/Google Drive/Algotrading NEW/Algotrading/ABEV3201701.csv"

# Load dataset
stocks_data = numpy.loadtxt(DATASET_PATH, delimiter=',', skiprows = 1, usecols = 1)

# Classifying dataset
classified_input = []
for x in range (stocks_data.size):
    if x > 0:
        if stocks_data[x] > stocks_data[x-1]:
            classified_input.append(1)
        else:
            classified_input.append(0)

# Splitting data
train_x, test_x, train_y, test_y = train_test_split(classified_input, classified_input, train_size = 0.7, test_size = 0.3)

# 2-class logistic regression
model = Sequential()
model.add(Dense(1, activation = 'sigmoid', input_dim = 1))
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(test_x, test_y, epochs = 10, validation_data = (train_x, train_y))
score = model.evaluate(test_x, test_y, verbose = 0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
