# NeuroMind

NeuroMind is an AI-enhanced real-time fMRI neurofeedback system designed to combat clinical depression.

## Overview

Depression affects over 280 million people worldwide, with many unable to receive the care they need due to barriers such as access, stigma, and unpredictable treatment outcomes. NeuroMind aims to address this pressing issue by combining cutting-edge technology with mental health care.

## Features

- AI-enhanced real-time fMRI neurofeedback (rt-fMRI-NF)
- Personalized therapy option
- Non-invasive treatment
- Precise targeting of brain regions
- Real-time feedback and adjustments
- User-friendly interface for clinicians and patients

## Technology Stack

- **Data Source**: OpenNeuro dataset
- **Data Preprocessing**: Python, NiLearn, NiBabel
- **AI Model**: 3D Convolutional Neural Network, Multi-layer Perceptron
- **Backend**: Python, Django
- **Frontend**: JavaScript, React

## Development Process

1. Data sourcing and preprocessing
2. AI model development and training
3. Web application development
4. Extensive testing and validation
5. Real-time data processing pipeline implementation

## Iterations of Model 
0. Proof of concept
- First we tested the concept with a pretrained Py-Torch model
- This model saw limited success and couldn't converage on adequate weights to predict valence and arousal but had a fair progression of loss showing potential in the model 
- The mean square errors (MSE) for valence and arousal were 5.313 and 4.124 respectivley following 50 epochs of training
  
   ```python
   import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision.models.video import r3d_18
    from torch.utils.data import DataLoader, Dataset
    import nibabel as nib
    import numpy as np
    import pandas as pd
   
   # Custom Dataset class for loading fMRI data
    class FMRIDataset(Dataset):
        def __init__(self, fmri_paths, event_paths, transform=None):
            self.fmri_paths = fmri_paths
            self.event_paths = event_paths
            self.transform = transform
        
        def __len__(self):
            return len(self.fmri_paths)
        
        def __getitem__(self, idx):
            fmri_path = self.fmri_paths[idx]
            event_path = self.event_paths[idx]
            
            fmri_img = nib.load(fmri_path).get_fdata()
            event_df = pd.read_csv(event_path, sep='\t')
            event_df = event_df.dropna().reset_index(drop=True)
            
            arousal = event_df['arousal']
            arousal = (arousal - np.min(arousal)) / (np.max(arousal) - np.min(arousal))
            valence = event_df['valence']
            valence = (valence - np.min(valence)) / (np.max(valence) - np.min(valence))
            labels = np.stack([arousal, valence], axis=1)
            
            fmri_img = torch.tensor(fmri_img, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.float32)
            
            if self.transform:
                fmri_img = self.transform(fmri_img)
            
            return fmri_img, labels
    
    # Define the 3D convolutional model with pre-trained ResNet
    class Conv3DModel(nn.Module):
        def __init__(self):
            super(Conv3DModel, self).__init__()
            self.base_model = r3d_18(pretrained=True)
            self.base_model.fc = nn.Identity()  # Remove the original classifier
            self.fc1 = nn.Linear(512, 1)  # For arousal
            self.fc2 = nn.Linear(512, 1)  # For valence
        
        def forward(self, x):
            features = self.base_model(x)
            arousal = self.fc1(features)
            valence = self.fc2(features)
            return arousal, valence
    
    # Prepare the data
    fmri_paths = [f"data/sub-{i:03d}/func/sub-{i:03d}_task-rest_bold.nii.gz" for i in range(1, 4)]
    event_paths = [f"data/sub-{i:03d}/func/sub-{i:03d}_task-rest_events.tsv" for i in range(1, 4)]
    
    dataset = FMRIDataset(fmri_paths, event_paths)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize the model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Conv3DModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs_arousal, outputs_valence = model(inputs)
            loss_arousal = criterion(outputs_arousal.squeeze(), labels[:, 0])
            loss_valence = criterion(outputs_valence.squeeze(), labels[:, 1])
            loss = loss_arousal + loss_valence
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')
    
    print('Training complete')
   ```
   
1. Version One of Bespoke Model
- This inital model followed our general outline of architecture but included shallow layers
- We choose to start simple and build up inorder to scale to our hardware capabillities and see if a bespoke model can adequatley predict values
- Folowing 50 epochs of training, the best checkpoint from this iteration yeilded (Mean Absolute Error) MAE's of 4.571 and 3.712 with consistently tapering loss
- However, this model was still fairly naive as it often predicted the mean with slight alterations


```python
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, BatchNormalization, Dropout, Input
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint

def get_compiled_model():
    inputs = Input((80, 80, 37, 1))
        
    # Convolutional layers with batch normalization
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='elu')(x)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    # x = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='elu')(x)
    # x = BatchNormalization()(x)
    x = Flatten()(x)
    
    # Arousal Branch
    arousal_x = Dense(units=16, activation='elu', kernel_regularizer=l2(0.01))(x)
    arousal_x = Dropout(0.2)(arousal_x)
    arousal_x = Dense(units=4, activation='elu', kernel_regularizer=l2(0.01))(arousal_x)
    out_arousal = Dense(units=1, activation='linear', name='norm_arousal')(arousal_x)
    
    # Valence Branch
    valence_x = Dense(units=64, activation='elu', kernel_regularizer=l2(0.01))(x)
    valence_x = Dropout(0.2)(valence_x)
    valence_x = Dense(units=4, activation='elu', kernel_regularizer=l2(0.01))(valence_x)
    out_valence = Dense(units=1, activation='linear', name='norm_valence')(valence_x)
    
    # Define the model with multiple outputs
    model = Model(inputs=inputs, outputs=[out_arousal, out_valence])
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=3),
                  loss=['mse', 'mse'],
                  loss_weights=[1, 1],
                  metrics=['mae', 'mae'])
    
    return model
```

2. Version Two of Bespoke Model
- This second model included more dense and convelututions layers, and showed subsatinal improvemnt from the prior. We also added more trainable parameters.
- Folowing 50 epochs of training, the best checkpoint from this iteration yeilded MAE's of 1.620 and 2.918 with consistently tapering loss
- However this mdoel was a lot more computationaly exahustive and requried use to utilize more gpus for training (2 x NVIDIA 1080ti's)


```python
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, BatchNormalization, Dropout, Input
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint

def get_compiled_model():
    inputs = Input((80, 80, 37, 1))
        
    # Convolutional layers with batch normalization
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='elu')(x)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    # x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='elu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    # x = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='elu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)

    x = Flatten()(x)
    
    # Arousal Branch
    arousal_x = Dense(units=64, activation='elu', kernel_regularizer=l2(0.01))(x)
    arousal_x = Dropout(0.2)(arousal_x)
    arousal_x = Dense(units=32, activation='elu', kernel_regularizer=l2(0.01))(arousal_x)
    arousal_x = Dropout(0.2)(arousal_x)
    arousal_x = Dense(units=16, activation='elu', kernel_regularizer=l2(0.01))(arousal_x)
    arousal_x = Dropout(0.2)(arousal_x)
    arousal_x = Dense(units=8, activation='elu', kernel_regularizer=l2(0.01))(arousal_x)
    arousal_x = Dropout(0.2)(arousal_x)
    arousal_x = Dense(units=4, activation='elu', kernel_regularizer=l2(0.01))(arousal_x)
    out_arousal = Dense(units=1, activation='linear', name='norm_arousal')(arousal_x)
    
    # Valence Branch
    valence_x = Dense(units=64, activation='elu', kernel_regularizer=l2(0.01))(x)
    valence_x = Dropout(0.2)(valence_x)
    valence_x = Dense(units=32, activation='elu', kernel_regularizer=l2(0.01))(valence_x)
    valence_x = Dropout(0.2)(valence_x)
    valence_x = Dense(units=8, activation='elu', kernel_regularizer=l2(0.01))(valence_x)
    valence_x = Dropout(0.2)(valence_x)
    valence_x = Dense(units=4, activation='elu', kernel_regularizer=l2(0.01))(valence_x)
    out_valence = Dense(units=1, activation='linear', name='norm_valence')(valence_x)
    
    # Define the model with multiple outputs
    model = Model(inputs=inputs, outputs=[out_arousal, out_valence])
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=3),
                  loss=['mse', 'mse'],
                  loss_weights=[1, 1],
                  metrics=['mae', 'mae'])
    
    return model
```

3. Beta Neuro Mind Model
- This model is fairly simillar to the current model however the code is a bit redundant which we streamlined in our final iteration.
- We changed the loss from MSE to MAE, changed our optimizer to RMSprop, and increased clipnorm to 5 and we found these alterations benifical to training
- Folowing 85 epochs of training, the best checkpoint from this iteration yeilded MAE's of 0.252 and 0.216 with consistently tapering loss

```python
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, BatchNormalization, Dropout, Input
from keras.models import Model
from keras.regularizers import l2
from keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

# Lone Relu layer to remove brain mass
def filter_layer(inputs):
  return tf.keras.activations.relu(inputs)
    
def conv_block(x, filters, kernel_size, activation, kernel_initializer, kernel_regularizer):
    x = Conv3D(filters=filters, kernel_size=kernel_size, activation=activation, kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    return x

def dense_block(x, units, activation, kernel_initializer, kernel_regularizer):
    x = Dense(units=units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    x = Dropout(0.2)(x)
    return x

def get_compiled_model(arou_kernal_reg=0.001, vale_kernal_reg=0.001, clipnorm=0):
    inputs = Input((80, 80, 37, 1))
    print(f"Input shape: {inputs.shape}")

    # Filter Layer
    x = filter_layer(inputs)
    print(f"Filter layer shape: {x.shape}")

    intil = 'he_normal'
    # Convolutional layers with batch normalization
    x = conv_block(x, filters=128, kernel_size=(3, 3, 3), activation='elu', kernel_initializer=intil, kernel_regularizer=None)
    x = conv_block(x, filters=128, kernel_size=(3, 3, 3), activation='elu', kernel_initializer=intil, kernel_regularizer=None)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    print(f"Max pool layer 1 shape: {x.shape}")
    x = conv_block(x, filters=64, kernel_size=(3, 3, 3), activation='elu', kernel_initializer=intil, kernel_regularizer=None)
    x = conv_block(x, filters=64, kernel_size=(3, 3, 3), activation='elu', kernel_initializer=intil, kernel_regularizer=None)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    print(f"Max pool layer 2 shape: {x.shape}")
    x = Flatten()(x)
    print(f"Flatten layer shape: {x.shape}")

    # Arousal Branch
    arousal_x = dense_block(x, units=512, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    arousal_x = dense_block(arousal_x, units=256, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    arousal_x = dense_block(arousal_x, units=128, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    arousal_x = dense_block(arousal_x, units=64, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    arousal_x = dense_block(arousal_x, units=32, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    arousal_x = dense_block(arousal_x, units=16, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    arousal_x = dense_block(arousal_x, units=8, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    out_arousal = Dense(units=1, activation='linear', name='arousal')(arousal_x)
    print(f"Arousal output shape: {out_arousal.shape}")

    # Valence Branch
    valence_x = dense_block(x, units=512, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    valence_x = dense_block(valence_x, units=256, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    valence_x = dense_block(valence_x, units=128, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    valence_x = dense_block(valence_x, units=64, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    valence_x = dense_block(valence_x, units=32, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    valence_x = dense_block(valence_x, units=16, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    valence_x = dense_block(valence_x, units=8, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    out_valence = Dense(units=1, activation='linear', name='valence')(valence_x)
    print(f"Valence output shape: {out_valence.shape}")

    # Define the model with multiple outputs
    model = Model(inputs=inputs, outputs=[out_arousal, out_valence])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, clipnorm=clipnorm),
                  loss=['mae', 'mae'],
                  loss_weights=[1, 1],
                  metrics=['mae','mae'])

    return model
```

4. Final Neuro Mind Model
- This is our final model

```python
@tf.function
def custom_loss(y_true, y_pred):
    # Define the range of acceptable values
    min_val = 0.0
    max_val = 1.0

    # Calculate the penalty for values outside of the range
    penalty = tf.where(y_pred > max_val, y_pred - max_val, 0.0) + tf.where(y_pred < min_val, min_val - y_pred, 0.0)

    # Calculate the mean absolute error loss
    loss = tf.keras.losses.MeanAbsoluteError()

    # Add the penalty to the mean absolute error loss
    total_loss = loss.call(y_true=y_true,y_pred=y_pred) + penalty

    return total_loss

    
# Lone Relu layer to remove brain mass
def filter_layer(inputs):
  return tf.keras.activations.relu(inputs)
    
def conv_block(x, filters, kernel_size, activation, kernel_initializer, kernel_regularizer):
    x = Conv3D(filters=filters, kernel_size=kernel_size, activation=activation, kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    return x

def dense_block(x, units, activation, kernel_initializer, kernel_regularizer):
    x = Dense(units=units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    x = Dropout(0.2)(x)
    return x

def get_compiled_model(arou_kernal_reg=0.001, vale_kernal_reg=0.001, clipnorm=0):
    inputs = Input((80, 80, 37, 1))
    print(f"Input shape: {inputs.shape}")

    # Filter Layer
    x = filter_layer(inputs)
    print(f"Filter layer shape: {x.shape}")

    intil = 'he_normal'
    # Convolutional layers with batch normalization
    x = conv_block(x, filters=256, kernel_size=(3, 3, 3), activation='elu', kernel_initializer=intil, kernel_regularizer=None)
    x = conv_block(x, filters=128, kernel_size=(3, 3, 3), activation='elu', kernel_initializer=intil, kernel_regularizer=None)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    print(f"Max pool layer 1 shape: {x.shape}")
    x = conv_block(x, filters=64, kernel_size=(3, 3, 3), activation='elu', kernel_initializer=intil, kernel_regularizer=None)
    x = conv_block(x, filters=32, kernel_size=(3, 3, 3), activation='elu', kernel_initializer=intil, kernel_regularizer=None)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    print(f"Max pool layer 2 shape: {x.shape}")
    x = Flatten()(x)
    print(f"Flatten layer shape: {x.shape}")

    # Arousal Branch
    arousal_x = dense_block(x, units=512, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    arousal_x = dense_block(arousal_x, units=256, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    arousal_x = dense_block(arousal_x, units=128, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    arousal_x = dense_block(arousal_x, units=64, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    arousal_x = dense_block(arousal_x, units=32, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    arousal_x = dense_block(arousal_x, units=16, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    arousal_x = dense_block(arousal_x, units=8, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(arou_kernal_reg))
    out_arousal = Dense(units=1, activation='linear', name='arousal')(arousal_x)
    print(f"Arousal output shape: {out_arousal.shape}")

    # Valence Branch
    valence_x = dense_block(x, units=512, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    valence_x = dense_block(valence_x, units=256, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    valence_x = dense_block(valence_x, units=128, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    valence_x = dense_block(valence_x, units=64, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    valence_x = dense_block(valence_x, units=32, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    valence_x = dense_block(valence_x, units=16, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    valence_x = dense_block(valence_x, units=8, activation='elu', kernel_initializer=intil, kernel_regularizer=l2(vale_kernal_reg))
    out_valence = Dense(units=1, activation='linear', name='valence')(valence_x)
    print(f"Valence output shape: {out_valence.shape}")

    # Define the model with multiple outputs
    model = Model(inputs=inputs, outputs=[out_arousal, out_valence])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, clipnorm=clipnorm),
                  loss=[custom_loss, custom_loss],
                  loss_weights=[1, 1],
                  metrics=['mae','mae'])

    return model

```
## Impact

NeuroMind aims to improve treatment outcomes for individuals suffering from depression by offering precise and personalized neurofeedback. The project has the potential to revolutionize mental health care, making it more accessible, personalized, and effective.

## Future Plans

- Expand to clinical trials
- Bring the solution to a broader audience


## Get Involved

We're excited to share more about how NeuroMind can profoundly impact mental health care. For questions or collaboration opportunities, please contact the team.

Together, we can create a future where mental health care is accessible, deeply personalized, and effective for everyone.
