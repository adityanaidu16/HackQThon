import numpy as np
import pandas as pd
import tensorflow as tf
from qiskit_finance import QiskitFinanceError
from qiskit_finance.data_providers import *
import datetime
# External imports
from pylab import cm
import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import circuit_drawer
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.algorithms import QSVR
from qiskit_machine_learning.datasets import ad_hoc_data


# In[5]:


from qiskit_finance import QiskitFinanceError
from qiskit_finance.data_providers import *

def find_predicted_price(stock_symbol):
            #!/usr/bin/env python
    # coding: utf-8

    # In[1]:


    get_ipython().system('pip install qiskit_finance')


    # In[1]:


    from qiskit import IBMQ


    # In[2]:


    IBMQ.save_account('72a2b314d20f97dd10cee423be6dbae1fe8a2b8a5b6181531c77501c2ca689cae045d13f9ea289e47968faede9fdd55a7a1e22e90725aada68dc2348f67ffe66')


    # In[3]:


    IBMQ.load_account()


    # In[ ]:





    # Loading main data

    # In[4]:
    dt = YahooDataProvider(tickers= "SPY",
                     start = datetime.datetime(2016, 1, 1),
                     end = datetime.datetime(2022, 5, 30))
    dt.run()
    data = pd.DataFrame(dt._data)
    data = data.T
    data.head(3)


    # In[6]:


    # Sort the data points based on indexes just for confirmation 
    data.sort_index(inplace = True)


    # In[7]:


    # Remove any duplicate index 
    data = data.loc[~data.index.duplicated(keep='first')]


    # In[8]:


    data.tail(3)


    # In[9]:


    data.head()


    # In[10]:


    # Check for missing values 
    data.isnull().sum()


    # In[11]:


    # Get the statistics of the data
    data.describe()


    # Understanding Trends with in the Data

    # In[12]:


    get_ipython().system('pip install plotly')
    import plotly.graph_objects as go


    # In[13]:


    # Check the trend in Closing Values 
    data['Adj Close'].plot(figsize=(16,6))


    # Data Preparation

    # In[14]:


    from sklearn.preprocessing import MinMaxScaler 
    import pickle 
    from tqdm.notebook import tnrange


    # In[15]:


    # Filter only required data 
    data = data[['Adj Close']]
    data.head(3)


    # Scrapping extra information

    # In[16]:


    import requests 

    response = requests.get('https://www.alphavantage.co/query?function=RSI&symbol=GOOGL&interval=daily&time_period=5&series_type=close&apikey=43T9T17VCV2ME4SM') 
    response = response.json()


    # In[17]:


    response.keys()


    # In[18]:


    rsi_data = pd.DataFrame.from_dict(response['Technical Analysis: RSI'] , orient='index')


    # In[19]:


    rsi_data.head()


    # In[20]:


    rsi_data = rsi_data[rsi_data.index >= '2018-01-01']


    # In[21]:


    rsi_data['RSI'] = rsi_data['RSI'].astype(np.float64)


    # In[22]:


    rsi_data.head()


    # In[23]:


    data = data.merge(rsi_data, left_index=True, right_index=True, how='inner')


    # In[24]:


    data.head()


    # In[25]:


    # Confirm the Testing Set length 
    test_length = data[(data.index >= '2020-09-01')].shape[0]


    # In[26]:


    def CreateFeatures_and_Targets(data, feature_length):
        X = []
        Y = []

        for i in tnrange(len(data) - feature_length): 
            X.append(data.iloc[i : i + feature_length,:].values)
            Y.append(data["Adj Close"].values[i+feature_length])

        X = np.array(X)
        Y = np.array(Y)

        return X , Y


    # In[27]:


    X , Y = CreateFeatures_and_Targets(data , 32)


    # In[ ]:





    # In[28]:


    # Check the shapes
    X.shape , Y.shape


    # In[29]:


    Xtrain , Xtest , Ytrain , Ytest = X[:-test_length] , X[-test_length:] , Y[:-test_length] , Y[-test_length:]


    # In[30]:


    # Check Training Dataset Shape 
    Xtrain.shape , Ytrain.shape


    # In[31]:


    # Check Testing Dataset Shape
    Xtest.shape , Ytest.shape


    # In[32]:


    from qiskit_machine_learning.algorithms import VQC
    # import the feature map and ansatz circuits
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    # import the optimizer for the training
    from qiskit.algorithms.optimizers import L_BFGS_B
    # import backend
    from qiskit.providers.aer import QasmSimulator


    # In[33]:


    num_qubits = 2
    vqc = VQC(feature_map=ZZFeatureMap(num_qubits), 
              ansatz=RealAmplitudes(num_qubits, reps=1), 
              loss='cross_entropy', 
              optimizer=L_BFGS_B(),
              quantum_instance=QasmSimulator())
    # train classifier
    vqc.fit(Xtrain, Ytrain)
    # score result
    vqc.score(Xtest, Ytest)


    # In[ ]:





    # In[ ]:


    VQC['Adj Close'].plot(figsize=(16,6)


    # In[ ]:
    return predicted_price


