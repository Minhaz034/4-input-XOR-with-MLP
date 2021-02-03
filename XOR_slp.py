import numpy as np
import pandas as pd
df  = pd.read_csv("4_input_XOR.csv")
df = df.sample(frac = 1)   

np.random.seed(0)

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sx):
   
    return sx * (1 - sx)


def cost(predicted, truth):
    return (truth - predicted)


X=df.iloc[:, :-1 ].to_numpy()
Y = df.iloc[:,-1].to_numpy()




#splitting train and test set:
train_split = int(len(X) / 100 * 60)
X_train = X[:train_split]
y_train = Y[:train_split]

X_test = X[train_split:]
y_test = Y[train_split:]





y_test = y_test.reshape(y_test.shape[0],1)
y_train = y_train.reshape(y_train.shape[0],1)
# Define the shape of the weight vector.
num_data, input_dim = X_train.shape
# Lets set the dimensions for the intermediate layer.
hidden_dim = 6
# Initialize weights between the input layers and the hidden layer.
W1 = np.random.random((input_dim, hidden_dim))

output_dim = 2
# Initialize weights between the hidden layers and the output layer.
W2 = np.random.random((hidden_dim, output_dim))

num_epochs = 100
learning_rate = 0.5

for epoch_n in range(num_epochs):
    layer0 = X_train
    m=X_train.shape[1]
       
    layer1 = sigmoid(np.dot(layer0, W1))
    layer2 = sigmoid(np.dot(layer1, W2))

    
    layer2_error = cost(layer2, y_train)
    
    layer2_delta = layer2_error * sigmoid_derivative(layer2)

    layer1_error = np.dot(layer2_delta, W2.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)
 
    W2 +=  learning_rate * np.dot(layer1.T, layer2_delta)
    
    W1 +=  learning_rate * np.dot(layer0.T, layer1_delta)    
    
    
prediction=[]    
print(" Input     prediction    actual")    
for x, y in zip(X_test, y_test):
    layer1_prediction = sigmoid(np.dot(W1.T, x)) # Feed the unseen input into trained W.
    prediction = layer2_prediction = sigmoid(np.dot(W2.T, layer1_prediction)) # Feed the unseen input into trained W.
  
    print(x,"    ", np.argmax(prediction),"       ", y)     
  

  
