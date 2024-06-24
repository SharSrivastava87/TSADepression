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
1. Version One of the Model
2. 
```python
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, BatchNormalization, Dropout, Input
from keras.models import Model
from keras.regularizers import l2
from keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras.callbacks import ModelCheckpoint

def get_compiled_model():
    inputs = Input((80, 80, 37, 1))
    
    # Filter Layer
    x = filter_layer(inputs)
    
    # Convolutional layers with batch normalization
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='elu')(x)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='elu')(x)
    x = BatchNormalization()(x)
    # x = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='elu')(x)
    # x = BatchNormalization()(x)
    x = Flatten()(x)
    
    # Arousal Branch
    arousal_x = Dense(units=16, activation='elu', kernel_regularizer=l2(0.01))(x)
    arousal_x = Dropout(0.2)(arousal_x)
    arousal_x = Dense(units=8, activation='elu', kernel_regularizer=l2(0.01))(arousal_x)
    arousal_x = Dropout(0.4)(arousal_x)
    arousal_x = Dense(units=4, activation='elu', kernel_regularizer=l2(0.01))(arousal_x)
    out_arousal = Dense(units=1, activation='linear', name='norm_arousal')(arousal_x)
    
    # Valence Branch
    valence_x = Dense(units=64, activation='elu', kernel_regularizer=l2(0.01))(x)
    valence_x = Dropout(0.2)(valence_x)
    valence_x = Dense(units=8, activation='elu', kernel_regularizer=l2(0.01))(valence_x)
    valence_x = Dropout(0.3)(valence_x)
    valence_x = Dense(units=4, activation='elu', kernel_regularizer=l2(0.01))(valence_x)
    out_valence = Dense(units=1, activation='linear', name='norm_valence')(valence_x)
    
    # Define the model with multiple outputs
    model = Model(inputs=inputs, outputs=[out_arousal, out_valence])
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, clipnorm=3),
                  loss=['mse', 'mse'],
                  loss_weights=[1, 1],
                  metrics=[MeanAbsolutePercentageError(), MeanAbsolutePercentageError()])
    
    return model
```
## Impact

NeuroMind aims to improve treatment outcomes for individuals suffering from depression by offering precise and personalized neurofeedback. The project has the potential to revolutionize mental health care, making it more accessible, personalized, and effective.

## Future Plans

- Expand to clinical trials
- Bring the solution to a broader audience

## Contributors

- Sanmay
- Sharvay
- Siddharth

## Get Involved

We're excited to share more about how NeuroMind can profoundly impact mental health care. For questions or collaboration opportunities, please contact the team.

Together, we can create a future where mental health care is accessible, deeply personalized, and effective for everyone.
