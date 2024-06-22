# Import required libraries


```python
import numpy as np
import pandas as pd
```

# Load Dataset
We'll be using iris-flower-dataset, aviable publicly in kaggle or github repo [here](https://gist.github.com/curran/a08a1080b88344b0c8a7)


```python
dataset_link = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
df = pd.read_csv(dataset_link)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



Let's not discuss much about dataset, as we've used it a lot in other simple projects as well.


```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])
```


```python
df.sample(5)
```





<div class="df">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65</th>
      <td>6.7</td>
      <td>3.1</td>
      <td>4.4</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>107</th>
      <td>7.3</td>
      <td>2.9</td>
      <td>6.3</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>50</th>
      <td>7.0</td>
      <td>3.2</td>
      <td>4.7</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>6.8</td>
      <td>2.8</td>
      <td>4.8</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>5.2</td>
      <td>3.4</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('species', axis=1), df['species'], test_size=0.2)
```


```python
import torch

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
```

# Building structure of network


```python
# First import pytorch, its nn and optim module.

import torch
import torch.nn as nn
import torch.optim as optim
```

To create neural networks, we need to create a child class of nn.Module, define it's layer structure in constructor and create a forward function that tells how forward propagation works. <br>
We can also define layers in different other ways as well, but for this project, let's use simplest way. <br>
In forward pass we pass x through all of the fully connected layers plus pass the result of each layer through relu activation function for non linearity.


```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define 3 fully connected layers
        self.fc1 = nn.Linear(4, 128)  # Input layer (4 features) to hidden layer
        self.fc2 = nn.Linear(128, 64) # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(64, 3)   # Hidden layer to output layer (3 classes)
        self.relu = nn.ReLU()         # Activation function

    def forward(self, x):
        # Define forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```

# Training our model

Before we start to train our model, we need to define criterion or loss function and an optimizer. In this case we'll be using Cross entropy loss and Adam optimizer respectively.


```python
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```


```python
batch_size = 16
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if((epoch + 1) % 10 == 0):
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


```


*Epoch [10/100], Loss: 0.1284* <br>
*Epoch [20/100], Loss: 0.0169* <br>
*Epoch [30/100], Loss: 0.0047* <br>
*Epoch [40/100], Loss: 0.0021* <br>
*Epoch [50/100], Loss: 0.0012* <br>
*Epoch [60/100], Loss: 0.0008* <br>
*Epoch [70/100], Loss: 0.0006* <br>
*Epoch [80/100], Loss: 0.0004* <br>
*Epoch [90/100], Loss: 0.0003* <br>
*Epoch [100/100], Loss: 0.0003* <br>


# Model evaluation


```python
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(0, len(X_test_tensor), batch_size):
        X_batch = X_test_tensor[i:i+batch_size]
        y_batch = y_test_tensor[i:i+batch_size]
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')
```


*Accuracy of the model on the test set: 100.00%*


# Prediction from model


```python
def predict_class(input_data):
    # Convert input data to tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
    
    return predicted.item()

user_input = [6.7, 3.3, 5.6, 2.2]  

predicted_class = predict_class(user_input)
print(f'The predicted class for the input {user_input} is: {predicted_class}')
```


*The predicted class for the input [6.7, 3.3, 5.6, 2.2] is: 2*

