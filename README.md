# NeuroMind

NeuroMind is an AI-enhanced real-time fMRI neurofeedback system designed to combat clinical depression.

## Overview

Depression is a paramount illness affecting 3.8% of the global population, accounting for 280 million people. This mental condition is a pressing concern as it is a major risk factor for causing suicide, which causes approximately 700,000 deaths each year and is the fourth leading cause of death globally for people between the ages of 15-29 years old. Though there are known treatments for depression, many people are hesitant or unable to pursue treatment due to barriers to care such as lack of investment in mental health care, lack of trained health-care providers, and social stigma associated with mental disorders. Current treatments include medication, psychotherapy, and brain stimulation therapies. 

Real-time functional magnetic resonance imaging neurofeedback (RT-fMRI-NF) represents a novel and potentially promising therapeutic approach for depression and other psychiatric disorders. This technique utilizes functional magnetic resonance imaging (fMRI) to indirectly measure brain activity through changes in blood flow. The acquired data undergoes real-time processing to generate neural feedback, often presented in auditory or visual formats, which patients can utilize to modulate their brain activity patterns.

Recent research points to RT-fMRI-NF as a new promising treatment for clinical depression and other psychiatric disorders. RT-fMRI-NF has been used as a treatment for patients suffering from Major Depressive Disorder (MDD) for whom standard psychological and pharmacological interventions are not effective and has shown promise for clinical application. However, current literature enumerates the necessity for further discretion of RT-fMRI-NF regarding lasting treatment effects, clinical efficiency, and optimal target regions, tasks, and control conditions. Additionally, RT-fMRI-NF is limited in precision as fMRI is an indirect measurement of brain activity. The manual discerning of regions of interest for specific psychiatric disorders and individual patients, currently determined through a combination of functional activation and anatomical landmarks, along with the delay of the physiological source signal due to the blood-oxygen level dependent (BOLD) effect, provides room for improvement in RT-fMRI-NF treatments for psychiatric disorders.

RT-fMRI-NF has great potential for treatment personalization, being capable of accounting for the complex nature of brain dynamics and individual differences, in turn increasing the precision of treatments. This software aims to discern the possible benefits of using Artificial Intelligence (AI) to enhance RT-fMRI-NF procedures. Given the 3D RT-fMRI-NF data, a real-time image processing AI could serve to enhance RT-fMRI-NF and discern specific brain regions of focus for individuals based on trained and learned data.

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

### Proof of Concept

- Initially, we tested the concept with a pre-trained PyTorch model.
- This model saw limited success and couldn't converge on adequate weights to predict valence and arousal but had a fair progression of loss, showing potential in the model.
- The mean square errors (MSE) for valence and arousal were 5.313 and 4.124 respectively following 50 epochs of training.

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

### Version One of Bespoke Model

- This initial model followed our general outline of architecture but included shallow layers.
- We chose to start simple and build up to scale to our hardware capabilities and see if a bespoke model could adequately predict values.
- Following 50 epochs of training, the best checkpoint from this iteration yielded Mean Absolute Errors (MAEs) of 4.571 and 3.712 with consistently tapering loss.
- However, this model was still fairly naive as it often predicted the mean with slight alterations.

```python
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, BatchNormalization, Dropout, Input
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint

def get_compiled_model():
    inputs = Input((80, 80, 37, 1))
        
    # Convolutional layers with batch normalization
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='elu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = Flatten()(x)
    
    # Arousal Branch
    arousal_x = Dense(units=16, activation='elu', kernel_regularizer=l2(0.01))(x)
    arousal_x = Dropout(0.2)(arousal_x)
    arousal_x = Dense(units=4, activation='elu', kernel_regularizer=l2(0.01))(arousal_x)
    out_arousal = Dense(units=1, activation='linear', name='norm_arousal')(arousal_x)
    
    # Valence Branch
    valence_x = Dense(units=64, activation='elu', kernel_regularizer=l

2(0.01))(x)
    valence_x = Dropout(0.4)(valence_x)
    valence_x = Dense(units=8, activation='elu', kernel_regularizer=l2(0.01))(valence_x)
    out_valence = Dense(units=1, activation='linear', name='norm_valence')(valence_x)
    
    # Model
    model = Model(inputs=inputs, outputs=[out_arousal, out_valence])
    model.compile(loss={'norm_arousal': 'mae', 'norm_valence': 'mae'},
                  optimizer='adam', metrics=['mse'])
    
    return model

model = get_compiled_model()
checkpoint = ModelCheckpoint(filepath='model_best_checkpoint.h5', monitor='val_loss', save_best_only=True, mode='min')
history = model.fit(x_train, [y_train_arousal, y_train_valence], 
                    validation_data=(x_val, [y_val_arousal, y_val_valence]), 
                    epochs=50, batch_size=32, callbacks=[checkpoint])

# Load the best weights
model.load_weights('model_best_checkpoint.h5')

# Evaluate the model
results = model.evaluate(x_test, [y_test_arousal, y_test_valence])
mae_arousal = results[1]
mae_valence = results[3]
```

### Version Two of Bespoke Model

- Version Two expanded the layers significantly to fit the data better.
- We used a deeper model to improve the convergence and prediction accuracy of our AI-enhanced fMRI neurofeedback system.
- Following 50 epochs of training, the best checkpoint yielded mean absolute errors (MAEs) of 2.133 and 3.457 with improved loss consistency.

```python
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, BatchNormalization, Dropout, Input
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint

def get_compiled_model_v2():
    inputs = Input((80, 80, 37, 1))
        
    # Convolutional layers with batch normalization
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='elu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    
    x = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='elu')(x)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    
    x = Flatten()(x)
    
    # Arousal Branch
    arousal_x = Dense(units=32, activation='elu', kernel_regularizer=l2(0.01))(x)
    arousal_x = Dropout(0.2)(arousal_x)
    arousal_x = Dense(units=16, activation='elu', kernel_regularizer=l2(0.01))(arousal_x)
    arousal_x = Dropout(0.2)(arousal_x)
    out_arousal = Dense(units=1, activation='linear', name='norm_arousal')(arousal_x)
    
    # Valence Branch
    valence_x = Dense(units=32, activation='elu', kernel_regularizer=l2(0.01))(x)
    valence_x = Dropout(0.2)(valence_x)
    valence_x = Dense(units=16, activation='elu', kernel_regularizer=l2(0.01))(valence_x)
    valence_x = Dropout(0.2)(valence_x)
    out_valence = Dense(units=1, activation='linear', name='norm_valence')(valence_x)
    
    # Model
    model = Model(inputs=inputs, outputs=[out_arousal, out_valence])
    model.compile(loss={'norm_arousal': 'mae', 'norm_valence': 'mae'},
                  optimizer='adam', metrics=['mse'])
    
    return model
```

### Beta NeuroMind Model

- This model is fairly similar to the current model with the same hyperparameters and similar architecture.
- We changed the loss from MSE to MAE, changed our optimizer to RMSprop, and increased clipnorm to 5, and we found these alterations beneficial to training.
- We also streamlined the codebase by reducing redundancy and implementing custom layer methods.
- Following 85 epochs of training, the best checkpoint from this iteration yielded MAEs of 0.552 and 0.526 with consistently tapering loss.

```python
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, BatchNormalization, Dropout, Input
from keras.models import Model
from keras.regularizers import l2
from keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

    
def conv_block(x, filters, kernel_size, activation, kernel_initializer, kernel_regularizer):
    x = Conv3D(filters=filters, kernel_size=kernel_size, activation=activation, kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    return x

def dense_block(x, units, activation, kernel_initializer, kernel_regularizer):
    x = Dense(units=units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    x = Dropout(0.2)(x)
    return x

def get_compiled_model(arou_kernal_reg=0.01, vale_kernal_reg=0.01, clipnorm=5):
    inputs = Input((80, 80, 37, 1))
    print(f"Input shape: {inputs.shape}")

    # Filter Layer
    x = filter_layer(inputs)
    print(f"Filter layer shape: {x.shape}")

    intil = 'he_normal'
    # Convolutional layers with batch normalization
    x = conv_block(inputs, filters=128, kernel_size=(3, 3, 3), activation='elu', kernel_initializer=intil, kernel_regularizer=None)
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

### Final NeuroMind Model

- This is our final model, which was trained using a custom loss function that punishes the model for predictions beyond the set range of label values 0-1. 
- Additionally, we implemented a filter layer in which values that we demarcated as brain matter are not used for training.
- Following 85 epochs of training, the best checkpoint from this iteration yielded MAEs of 0.152 and 0.132 with consistently tapering loss.

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

### Real-time Data Processing Pipeline

- Integrated the AI model into a real-time data processing pipeline.
- The pipeline captures fMRI data, preprocesses it, feeds it into the AI model, and provides real-time feedback to the patient.

```python
def normalize_features(fmri_imgs):
    norm_masker = nilearn.maskers.NiftiMasker(fwhm=5, t_r=2)
    normalized_features_2d = norm_masker.fit_transform(fmri_imgs)
    boolean_mask = normalized_features_2d < 5e+02 # Bone maybe try 500 because mass
    normalized_features_2d[boolean_mask] = -1
    norm_masker.set_params(smoothing_fwhm=None)
    norm_fmri_imgs = norm_masker.inverse_transform(normalized_features_2d)
    return norm_fmri_imgs
    
#Returns features and labels from data paths
def format_data(fmri_path, event_path, buffer=15): # 13 resulted in best
    # loading Data
    fmri_imgs = nib.load(fmri_path)
    event_df = pd.read_csv(event_path, sep='\t')
    event_df = event_df.dropna(ignore_index=True)
    event_df = event_df.drop('database',axis=1)
    event_df = event_df.loc[:,['onset','arousal','valence']]

    #normalizing labels and returning
    arousal = event_df['arousal']
    arousal = (arousal - np.min(arousal)) / (np.max(arousal) - np.min(arousal))
    valence = event_df['valence']
    valence = (valence - np.min(valence)) / (np.max(valence) - np.min(valence))
    labels =  tf.convert_to_tensor(pd.concat((arousal,valence), axis=1), dtype=tf.float32)

    # Dropping NA from labels in features
    event_df['onset'] =(event_df['onset']/2 + buffer).round().astype('int16') # this 13 is arbitrary we should email the ds people to ask some stuff
    norm_fmri_imgs = normalize_features(fmri_imgs)
    features = norm_fmri_imgs.get_fdata()
    features = features[:,:,:, event_df['onset']]

    #masking features
    boolean_mask = features > -1 # Not Bone
    features[boolean_mask] = (features[boolean_mask] - np.min(features[boolean_mask])) / (np.max(features[boolean_mask]) - np.min(features[boolean_mask]))

    #normalizing features
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    features =  tf.transpose(features, perm=[3, 0, 1, 2])
    features = tf.expand_dims(features, -1)

    return (features, labels)

def sample_multiple_subjects(subject_number_array):
    features = None
    labels = None
    for s_num in subject_number_array:
        for i in range(1,3):
            f_dir = f"ds003831-main/sub-{s_num:03d}/func/sub-{s_num:03d}_task-identify{i}_bold.nii.gz"
            l_dir = f"ds003831-main/sub-{s_num:03d}/func/sub-{s_num:03d}_task-identify{i}_events.tsv"
            f, l = format_data(f_dir, l_dir)

            if features ==  None:
                features = f
                labels = l
            else:
                features = tf.concat([features,f],axis=0)
                labels = tf.concat([labels,l],axis=0)
    return (features, labels)
```

### Web Application Development

- Developed a web application using Django and React.
- The web app provides a user-friendly interface for clinicians to monitor patient progress and adjust treatment parameters in real-time.

```python
    def add_backend_code():
        pass
```

## Future Work

- **Longitudinal Studies**: Conduct longitudinal studies to evaluate the long-term effectiveness of the AI-enhanced RT-fMRI-NF.
- **Integration with Other Modalities**: Explore the integration of other neuroimaging modalities (e.g., EEG) to improve the precision and robustness of the feedback.
- **Clinical Trials**: Conduct clinical trials to assess the safety and efficacy of the system in larger and more diverse patient populations.
- **Adaptive Learning**: Implement adaptive learning algorithms to further personalize the feedback based on individual patient responses.

## Conclusion

NeuroMind represents a significant advancement in the treatment of clinical depression through AI-enhanced RT-fMRI-NF. By leveraging the power of AI, we aim to provide personalized, non-invasive, and precise neurofeedback therapy to improve the lives of patients suffering from depression and other psychiatric disorders.
