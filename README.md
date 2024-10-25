# Neural Network Model for Alphabet Soup Funding Success Prediction

## Overview

This analysis aims to develop a deep learning model to predict the success of organizations applying for funding from Alphabet Soup. Utilizing a dataset of over 34,000 funded organizations, the model assists the nonprofit foundation in making data-driven funding allocation decisions. Success is defined by the effective use of granted funds, indicated by the `IS_SUCCESSFUL` variable.

The analysis employs a neural network to classify applicants based on various features reflecting their characteristics and operational contexts. This predictive capability is critical for optimizing funding decisions and ensuring resources are allocated to promising organizations.

## Results

### Data Preprocessing

**Target Variable:**
- `IS_SUCCESSFUL`: A binary variable indicating whether the funding was used effectively.

**Feature Variables:**
- `APPLICATION_TYPE`
- `AFFILIATION`
- `CLASSIFICATION`
- `USE_CASE`
- `ORGANIZATION`
- `STATUS`
- `INCOME_AMT`
- `SPECIAL_CONSIDERATIONS`
- `ASK_AMT`

**Variables to Remove:**
- `EIN` and `NAME`: These columns serve only as unique identifiers and do not contribute to the model's predictive capabilities. They were excluded from the feature set.

### Model Architecture

The neural network model consists of the following layers:
- **Input Layer:** 80 neurons, corresponding to the number of features in the dataset.
- **Hidden Layer 1:** 30 neurons with a ReLU activation function, allowing for non-linear relationships.
- **Output Layer:** 1 neuron with a sigmoid activation function, providing a binary output (successful or not).

This architecture aims to capture the complexities of the relationships between the features and the target while maintaining a manageable number of parameters to avoid overfitting.

### Model Performance

- **Accuracy:** The model achieved an accuracy of 72% on the test data, falling short of the target performance of 75%.

### Optimization Strategies

To enhance model performance, the following strategies were implemented:
- **Architectural Adjustments:** Varying the number of hidden layers and neurons.
- **Activation Functions:** Testing different activation functions (e.g., Leaky ReLU) for deeper layers.
- **Dropout Regularization:** Introducing dropout layers to minimize overfitting.
- **Hyperparameter Tuning:** Adjusting the learning rate and batch size for optimal convergence.
- **Increased Epochs:** Allowing the model more opportunities to learn from the training data.

## Summary

The deep learning model establishes a foundational predictive framework for assessing the success of funding applicants. While achieving a 72% accuracy is commendable, it does not meet the desired target of 75%, indicating that further optimization is necessary.

## Further Optimization

Refer to `AlphabetSoupCharity_Optimization.ipynb` for additional optimization steps:

**Target Variable:**
- `IS_SUCCESSFUL`: A binary variable indicating whether the funding was used effectively.

**Feature Variables:**
- `APPLICATION_TYPE`
- `AFFILIATION`
- `CLASSIFICATION`
- `USE_CASE`
- `ORGANIZATION`
- `STATUS`
- `INCOME_AMT`
- `SPECIAL_CONSIDERATIONS`
- `ASK_AMT`

**Variables to Remove:**
- `EIN`

1. **Neurons and Layers:** 
   - Dense Layer 1: 15 neurons (6,795 parameters).
   - Dense Layer 2: 30 neurons (480 parameters).
   - Dense Layer 3: 35 neurons (1,085 parameters).
   - Output Layer: 1 neuron (36 parameters).
   - Total Trainable Parameters: 8,396.

2. **Activation Functions:** 
   - Hidden layers use the ReLU activation function to handle non-linearity.
   - The output layer employs the sigmoid activation function for binary classification.

3. **Model Performance:** 
   - The current architecture yields a solid accuracy of 79.24%. Further enhancements may include:
     - **Tuning Neurons and Layers:** Adding more neurons or layers to capture complex patterns while balancing the risk of overfitting.
     - **Regularization Techniques:** Implementing dropout or L2 regularization to prevent overfitting.
     - **Learning Rate Adjustments:** Fine-tuning the learning rate for better model performance.
