# Neural Probabilistic Language Model (NPLM)

This notebook implements a character-level **Neural Probabilistic Language Model** using PyTorch. The model is trained on a dataset of names to generate new, unique sequences. The architecture is based on the foundational paper *[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)* (Bengio et al., 2003), adapted for character-level predictions. This repository also contains the notes i took from Andrej Karpathy's NLP lectures (check the Jupyter Notebooks), I was able to build this with Andrej's lecture so a big thanks to him.

##  Project Overview

This approach utilizes **distributed feature representations (embeddings)** for characters. This allows the model to learn similarity between characters in a dense vector space and generalize better to unseen contexts, You will see in the jupyter notebook (Model.ipynb) that vowels are grouped together by the model

### Key Features
- **Tokenization**: Character-level (27 tokens: 26 letters + `.` as a delimiter).
- **Embeddings**: Maps characters to a 10-dimensional vector space.
- **Context Window**: Uses the previous **3 characters** to predict the 4th.
- **Architecture**: A Multilayer Perceptron (MLP) with a `tanh` hidden layer.

##  Model Architecture

- **Input**: 3 characters (converted to integers).
- **Embedding Layer**: 27 x 10 (Lookup table).
- **Hidden Layer**: 
  - Input: 30 dimensions (3 chars x 10 dim).
  - Neurons: 200 units with `tanh` activation.
- **Output Layer**: 27 neurons (logits for the next character probability).
- **Parameters**: ~11,897 trainable parameters.

## Dataset & Splitting

The model is trained on `names.txt` containing **32,033** unique names.
- **Training Set (80%)**: Optimization of weights/biases.
- **Dev/Validation Set (10%)**: Hyperparameter tuning.
- **Test Set (10%)**: Final performance evaluation.

## Training Details

- **Optimizer**: Mini-batch Stochastic Gradient Descent (SGD).
- **Batch Size**: 32 examples.
- **Iterations**: 300,000 steps.
- **Loss Function**: Cross Entropy Loss (`F.cross_entropy`).
- **Learning Rate Schedule**:
  - Steps 0 - 100k: `0.1`
  - Steps 100k - 200k: `0.01`
  - Steps 200k+: `0.001`

## Results

| Dataset | Final Loss |
| :--- | :--- |
| **Train** | ~2.1910 |
| **Dev** | ~2.2698 |
| **Test** | ~2.2755 |

## Sample Generations
<img src="https://github.com/user-attachments/assets/d6ada02d-3718-4084-a41b-ee3af000ee6a" width="711" alt="Model generated names output" />


##  Repository Structure

Here is a guide to the files located in this repository and their purposes:

*   **`model.ipynb`**  
    The core of the project. This Jupyter Notebook contains the complete implementation, including:
    *   Data loading and preprocessing.
    *   Neural network architecture definition.
    *   Training loop with visualization.
    *   Inference code to generate new names.

*   **`names.txt`**  
    The dataset file. It contains **32,033** unique names (one per line) used to train the model.

*   **`requirements.txt`**  
    A list of Python dependencies required to run the code. Use this to install the necessary libraries (e.g., `torch (Version 2.9.1)`, `matplotlib (Version 3.10.8)`, `jupyter / notebook / ipykernel`).

*   **`A Neural probabilistic language model.pdf`**  
    The original research paper by Bengio et al. (2003). It serves as the theoretical reference for the architecture implemented in this project.
