from math import sqrt
import numpy as np

def main():

    import pandas as pd
    data = pd.read_csv("alldata.csv")

    y_arr = data['Price']
    x_arr = data['Sentiment']
    date_arr = data['Date']

    # normalize BTC price
    from sklearn.preprocessing import MinMaxScaler
    mmscaler = MinMaxScaler(feature_range=(0, 1))
    y_arr = mmscaler.fit_transform(y_arr.values.reshape(-1, 1))
    x_arr = x_arr.values.reshape(-1, 1)
    #print(y_arr)

    # select 80% as training data
    percent = input('Please enter the percentage you want to set as training data. \n'
                    'The percentage should be within [5,95]: ')
    percent = int(percent)
    if percent < 5:
        percent = input('Please enter greater than 5 %: ')
        percent = int(percent)
    if percent > 95:
        percent = input('Please enter smaller than 95 %: ')
        percent = int(percent)
    print("Note that the rest ",100-percent," % will be testing data.")
    # since the data is corresponding with date, it has to be in order

    # print(len(y_arr), len(x_arr))

    index_cut = int(len(y_arr)*(100-percent)/100)
    test_x = x_arr[0:index_cut]
    train_x = x_arr[index_cut:len(x_arr)]
    test_y = y_arr[0:index_cut]
    train_y = y_arr[index_cut:len(x_arr)]
    test_date = date_arr[0:index_cut]

    print("Training data size = ", len(train_y))
    print("Testing data size = ", len(test_x))
    print("Start process data")

    # process data to use [days] day prev [prev-price and this day's sentiment]
    day = 2
    Xtrain = []
    Ytrain = []
    for i in range(0, len(train_x)-day):
        # price
        y = train_y[i]
        Ytrain.append(y)
        # x = get 2 previous day's price
        x = train_y[i+1:i+1+day]
        # x = append today's sentiment score from reddit data
        x = x.tolist()
        x.append(train_x[i].tolist())
        Xtrain.append(x)

    train_x = np.array(Xtrain)
    train_y = np.array(Ytrain)
    print("Training input shape: ", train_x.shape)
    print("Training output shape: ", train_y.shape)

    Xtest = []
    Ytest = []
    for i in range(0, len(test_x)-day):
        # y = train_y[i + 2]
        y = test_y[i]
        Ytest.append(y)
        # x = get 2 previous day's price
        x = test_y[i+1:i+1+day]
        # x = append today's sentiment score from reddit data
        x = x.tolist()
        x.append(test_x[i].tolist())
        Xtest.append(x)
    test_x = np.array(Xtest)
    test_y = np.array(Ytest)
    #print(test_x.shape)
    #print(test_y.shape)

    print("Training the model...")
    # Model building:
    # LSTM Networks with
    #   3 layers
    #   50 nodes per layers
    #   loss = Mean Absolute Error
    #   optimizer = Adam Optimization
    import keras
    lstm = keras.models.Sequential()
    lstm.add(keras.layers.LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    lstm.add(keras.layers.LSTM(50))
    lstm.add(keras.layers.Dense(1))
    lstm.compile(loss='mae', optimizer='adam')

    # Model training:
    #   bagging with
    #   batch_size = 100
    #   epochs = 300
    re = lstm.fit(train_x, train_y, epochs=300, batch_size=100, validation_data=(test_x, test_y), verbose=0, shuffle=False)
    # print(re)

    # Model evaluating:
    from sklearn.metrics import mean_squared_error
    predict_y = lstm.predict(test_x)
    err = sqrt(mean_squared_error(test_y, predict_y))
    print('raw err: %.3f' % err)

    # Price re-convert evaluating:
    predict_y = mmscaler.inverse_transform(predict_y.reshape(-1, 1))
    test_y = mmscaler.inverse_transform(test_y.reshape(-1, 1))
    err = sqrt(mean_squared_error(test_y, predict_y))
    print('err: %.3f' % err)

    py = np.array(predict_y)
    y = np.array(test_y)
    test_date = np.array(test_date)

    # print("py.shape", py.reshape(-1).shape)
    # print("y.shape", y.reshape(-1).shape)
    # print("date before", test_date.shape)
    # print("date.shape", test_date[day:len(test_date)].shape)

    from datetime import datetime
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    date = []
    for s in test_date[0:len(test_date)-day]:
        s = str(s)
        d = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
        date.append(d)
    date = np.array(date)
    print("[May 01, 2018] predicted BTC price (USD) = ", predict_y[0])

    # plotting
    plt.plot(date.reshape(-1), py.reshape(-1), label="predict_y", color='green')
    plt.plot(date[1:-1].reshape(-1), y[1:-1].reshape(-1), label="test_y", color='red')
    red_patch = mpatches.Patch(color='red', label='Actual Price')
    green_patch = mpatches.Patch(color='green', label='Predicted Price')
    plt.legend(handles=[red_patch, green_patch])
    plt.ylabel('BTC Prices (USD)')
    plt.xlabel('Test dates')
    plt.show()




if __name__ == "__main__":
    main()
