import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler 
import pickle 
from tqdm.notebook import tnrange
import requests 

def find_predicted_price(stock_symbol):
    data = yf.download(stock_symbol , start = "2018-01-01" , interval = '1d')
    data.head(3)
    data.sort_index(inplace = True)
    data = data.loc[~data.index.duplicated(keep='first')]
    data.tail(3)
    data.head()
    data.isnull().sum()
    data.describe()
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = data.index , y = data['Close'] , mode = 'lines'))
    fig.update_layout(height = 500 , width = 900, 
                    xaxis_title='Date' , yaxis_title='Close')

    fig = go.Figure()

    fig.add_trace(go.Scatter(x = data.index , y = data['Volume'] , mode = 'lines'))
    fig.update_layout(height = 500 , width = 900, 
                    xaxis_title='Date' , yaxis_title='Volume')


    data = data[['Close' , 'Volume']]
    data.head(3)


    response = requests.get('https://www.alphavantage.co/query?function=RSI&symbol=GOOGL&interval=daily&time_period=5&series_type=close&apikey=43T9T17VCV2ME4SM') 
    response = response.json()


    response.keys()

    rsi_data = pd.DataFrame.from_dict(response['Technical Analysis: RSI'] , orient='index')

    rsi_data.head()

    rsi_data = rsi_data[rsi_data.index >= '2018-01-01']

    rsi_data['RSI'] = rsi_data['RSI'].astype(np.float64)

    rsi_data.head()

    data = data.merge(rsi_data, left_index=True, right_index=True, how='inner')

    data.head()

    # Confirm the Testing Set length 
    test_length = data[(data.index >= '2020-09-01')].shape[0]

    def CreateFeatures_and_Targets(data, feature_length):
        X = []
        Y = []

        for i in tnrange(len(data) - feature_length): 
            X.append(data.iloc[i : i + feature_length,:].values)
            Y.append(data["Close"].values[i+feature_length])

        X = np.array(X)
        Y = np.array(Y)

        return X , Y


    X , Y = CreateFeatures_and_Targets(data , 32)

    # Check the shapes
    X.shape , Y.shape


    Xtrain , Xtest , Ytrain , Ytest = X[:-test_length] , X[-test_length:] , Y[:-test_length] , Y[-test_length:]

    # Check Training Dataset Shape 
    Xtrain.shape , Ytrain.shape

    # Check Testing Dataset Shape
    Xtest.shape , Ytest.shape


    # Create a Scaler to Scale Vectors with Multiple Dimensions 
    class MultiDimensionScaler():
        def __init__(self):
            self.scalers = []

        def fit_transform(self , X):
            total_dims = X.shape[2]
            for i in range(total_dims):
                Scaler = MinMaxScaler()
                X[:, :, i] = Scaler.fit_transform(X[:,:,i])
                self.scalers.append(Scaler)
            return X

        def transform(self , X):
            for i in range(X.shape[2]):
                X[:, :, i] = self.scalers[i].transform(X[:,:,i])
            return X 

    Feature_Scaler = MultiDimensionScaler()
    Xtrain = Feature_Scaler.fit_transform(Xtrain)
    Xtest = Feature_Scaler.transform(Xtest)

    Target_Scaler = MinMaxScaler()
    Ytrain = Target_Scaler.fit_transform(Ytrain.reshape(-1,1))
    Ytest = Target_Scaler.transform(Ytest.reshape(-1,1))


    def save_object(obj , name : str):
        pickle_out = open(f"{name}.pck","wb")
        pickle.dump(obj, pickle_out)
        pickle_out.close()

    def load_object(name : str):
        pickle_in = open(f"{name}.pck","rb")
        data = pickle.load(pickle_in)
        return data


    # Save your objects for future purposes 
    save_object(Feature_Scaler , "Feature_Scaler")
    save_object(Target_Scaler , "Target_Scaler")


    # Model Building


    from tensorflow.keras.callbacks import ModelCheckpoint , ReduceLROnPlateau

    save_best = ModelCheckpoint("best_weights.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25,patience=5, min_lr=0.00001,verbose = 1)


    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense , Dropout , LSTM , Bidirectional , BatchNormalization

    model = Sequential()

    model.add(Bidirectional(LSTM(512 ,return_sequences=True , recurrent_dropout=0.1, input_shape=(32, 3))))
    model.add(LSTM(256 ,recurrent_dropout=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(64 , activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(32 , activation='elu'))
    model.add(Dense(1 , activation='linear'))



    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.002)
    model.compile(loss='mse', optimizer=optimizer)



    history = model.fit(Xtrain, Ytrain,
                epochs=1,
                batch_size = 1,
                verbose=1,
                shuffle=False ,
                validation_data=(Xtest , Ytest),
                callbacks=[reduce_lr , save_best])



    # Load the best weights
    model.load_weights("best_weights.h5")


    # Visualize prediction on Test Set


    Predictions = model.predict(Xtest)


    Predictions = Target_Scaler.inverse_transform(Predictions)
    Actual = Target_Scaler.inverse_transform(Ytest)


    Predictions.shape


    Predictions = np.squeeze(Predictions , axis = 1)
    Actual = np.squeeze(Actual , axis = 1)



    # Check the Predictions vs Actual
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = data.index[-test_length:] , y = Actual , mode = 'lines' , name='Actual'))
    fig.add_trace(go.Scatter(x = data.index[-test_length:] , y = Predictions , mode = 'lines' , name='Predicted'))
    #fig.show()


    # Visualize Prediction on whole data


    Total_features = np.concatenate((Xtrain , Xtest) , axis = 0)



    Total_Targets = np.concatenate((Ytrain , Ytest) , axis = 0)



    Predictions = model.predict(Total_features)


    Predictions = Target_Scaler.inverse_transform(Predictions)
    Actual = Target_Scaler.inverse_transform(Total_Targets)


    Predictions = np.squeeze(Predictions , axis = 1)
    Actual = np.squeeze(Actual , axis = 1)



    # Check the trend in Volume Traded
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = data.index , y = Actual , mode = 'lines' , name='Actual'))
    fig.add_trace(go.Scatter(x = data.index , y = Predictions , mode = 'lines' , name='Predicted'))


    # Save and Load the whole model
    model.save("Model.h5")
    loaded_model = tf.keras.models.load_model("Model.h5")


    # Realtime Prediction

    def PredictStockPrice(Model , DataFrame , PreviousDate , feature_length = 32):
        idx_location = DataFrame.index.get_loc(PreviousDate)
        Features = DataFrame.iloc[idx_location - feature_length : idx_location,:].values
        Features = np.expand_dims(Features , axis = 0)
        Features = Feature_Scaler.transform(Features)
        Prediction = Model.predict(Features)
        Prediction = Target_Scaler.inverse_transform(Prediction)
        return Prediction[0][0]



    predicted_price= PredictStockPrice(loaded_model , data , '2022-05-27')
    return predicted_price


