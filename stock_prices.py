import sys
import numpy as np
import matplotlib.pyplot as plotter
import pandas as pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler


def normalize_data(data):
    """
    Using min-max normalization to convert all values in range (0,1).
    :param data: 
    :return: 
    """
    rows = data.shape[0]
    cols = data.shape[1]
    max = np.amax(data, axis=0)
    min = np.amin(data, axis=0)

    for x in range(0, rows):
        for y in range(0, cols):
            temp = data[x, y]
            normalizedVal = (temp - min) / (max - min)
            data[x, y] = normalizedVal
    return data


def create_regression_model():
    """
    Since the outcome we are predicting is a continuous quantity i.e.
    a stock price therefore we are making a regression model.
    Rnn is a sequence of layers
    Now we will add layers on top of our regression model object
    - First is the input layer
    - Activation = sigmoid
    - units =  number of memory units (experiment with different numbers)
    - input shape =  Number of features = 1 that is the stock price at time T
    :return: regression_model
    """
    regression_model = Sequential()
    regression_model.add(LSTM(units=4,
                              activation='sigmoid',
                              input_shape=(None, 1)))
    # Adding the output layer
    # here units in the dense class object correspond to the dimension of the
    # output . Since the dimension of our output is 1 .(Stock price at time T+1)
    # so units =1
    regression_model.add(Dense(units=1))
    # Now we compile an RNN and define an optimizer. We set loss = mean
    # squared error because the loss metric we consider while training is the
    # Mean squared error. As we learnt during our regression class the mean
    # squared error is the square of the difference between the predicted price
    # and the real price we sum this error for all observations and knowing
    # this value helps optimize the predictions
    regression_model.compile(optimizer='adam', loss='mean_squared_error')
    print('Created regression model')
    return regression_model


def fit_regression_model(regression_model, train_1, train_2):
    """
    :param regression_model: our regression Model
    :param train_1: input    
    :param train_2: output
    """
    regression_model.fit(train_1, train_2, epochs=120)


def preprocess_data(training_set_path, testing_set_path):
    """
    Used to preprocess the data.
    :param training_set_path: The path to google_stock_prices_train.csv
    :param testing_set_path: The path to google_stock_prices_test.csv
    """
    print "Using training set: {0}".format(training_set_path)
    print "Using testing set: {0}".format(testing_set_path)
    train_data = pandas.read_csv(training_set_path)
    train_data = train_data.iloc[:, 1:2].values
    test_data = pandas.read_csv(testing_set_path)
    test_data = test_data.iloc[:, 1:2].values
    actual_stock_prices = test_data
    print('Printing length of the test data')
    print(len(test_data))
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    print('Normalized training data')
    print(train_data)
    train_1 = train_data[0:len(train_data) - 1]
    train_2 = train_data[1:len(train_data)]
    # Need to reshape the input because of the data types expected in the input
    # by keras. train_1 is now a 3d array with the first dimension
    # corresponding to the list of observations. The second dimension is the
    # timestamp and third is the features
    train_1 = np.reshape(train_1, (len(train_data) - 1, 1, 1))
    return test_data, train_1, train_2, scaler, actual_stock_prices


def visualize_prediction(actual_stock_prices, predicted_stock_price):
    """
    Used to visualize the predicted stock prices. 
    """
    plotter.plot(actual_stock_prices, color='red', label='Actual Stock Prices')
    plotter.plot(predicted_stock_price, color='blue', label='Predicted Stock '
                                                            'Prices')
    plotter.title('Google stock price prediction')
    plotter.xlabel('Time (Days)')
    plotter.ylabel('Stock Price ($)')
    plotter.legend()
    plotter.show()


def main():
    training_set_path = sys.argv[1]
    testing_set_path = sys.argv[2]
    print 'Preprocessing data...'
    (test_data, train_1, train_2,
     scaler, actual_stock_prices) = preprocess_data(training_set_path,
                                                    testing_set_path)
    regression_model = create_regression_model()
    fit_regression_model(regression_model, train_1, train_2)
    test_data = scaler.transform(test_data)
    print 'Printing Test data'
    print test_data
    test_data = np.reshape(test_data, (len(test_data), 1, 1))
    print type(test_data)
    predicted_stock_price = regression_model.predict(test_data)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    print(predicted_stock_price)
    visualize_prediction(actual_stock_prices, predicted_stock_price)


if __name__ == '__main__':
    main()
