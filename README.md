# Small LM - Math
Small language model training using publicly available datasets, for performing mathematical computations such as basic math, going up to calculus and ODE/PDE solutions.


# Current State

1. The model is trained on the GSM8K dataset, which contains 10000+ examples of math problems.
2. The model has a config file to change the model size, number of layers, heads, etc.
3. The MLX framework is used to train the model. This is a relatively small model, it is meant to be trained on Apple Silicon M-series chips.
4. The model is currently unable to perform any mathematical operations, it is only able to repeat the training data, albeit with a small degree of variation.
5. The model's future development will be in the `dev` branch. There will be an effort made to improve the following:
    - Datasets used for training
    - Model size
    - Number of layers, heads, etc.
    - Training loop
    - Evaluation metrics
 


# Contributing

If you would like to contribute to this project, please feel free to fork the repository and create a pull request with your changes and improvements. Please remember that the objective is to create a small, fast, and efficient model that can be trained on Apple Silicon M-series chips.





