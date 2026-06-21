
```markdown
# PyTorch Neural Architectures 

A comprehensive study and implementation of core Deep Learning architectures using PyTorch. This project evaluates how different neural network biases (None, Spatial, Temporal) adapt to different data structures (Tabular, Vision, Sequences).

Institution: EMSI (École Marocaine des Sciences de l'Ingénieur)  
Author: BOUDANE Taha  
Module: Deep Learning (2025-2026)

---

## Step-by-Step Setup & Installation

Follow these steps to get the project running on any machine. The scripts are designed to **automatically download all necessary datasets** upon their first execution, so no manual data downloading is required!


```

### Step 1: Create a Virtual Environment (Recommended)

To avoid conflicts with other Python projects, create and activate a virtual environment:

* **Windows:**
```bash
python -m venv env
.\env\Scripts\activate

```


* **Mac/Linux:**
```bash
python3 -m venv env
source env/bin/activate

```



### Step 2: Install Dependencies

Install all required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt

```

> **Note for NVIDIA GPU Users:** To take full advantage of CUDA acceleration, ensure you install the CUDA-specific version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before running the command above.

---

## Running the Project (The Interactive Dashboard)

To make navigation easy, this project features a master Command Line Interface (CLI) dashboard. You do not need to run the scripts individually.

Simply execute the main file:

```bash
python main.py

```

###  The Main Menu

Upon running `main.py`, your terminal will clear and present the custom EMSI dashboard. You simply type `1`, `2`, or `3` and press Enter to instantly launch the training and evaluation pipeline for that specific architecture.

<img width="1490" height="654" alt="image" src="https://github.com/user-attachments/assets/4081c888-b2b9-4ca8-a1e9-0bc7d3f2e736" />


---

##  Project Modules Breakdown

###  Part I: Tabular Classification (MLP)

**File:** `part1_mlp_tabular.py` | **Dataset:** Wine Quality (Physicochemical properties)

This module evaluates a Multi-Layer Perceptron (MLP) on structured tabular data. Since MLPs lack inductive bias, this section heavily explores the mathematical importance of weight initialization.

* Tests **Constant**, **Gaussian**, and **Xavier (Glorot)** initializations.
* Evaluates using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

<img width="1034" height="482" alt="image" src="https://github.com/user-attachments/assets/48267e5c-d3b4-40f7-8b73-647add1bf6e4" />


---

###  Part II: Computer Vision (CNN)

**File:** `part2_cnn_vision.py` | **Dataset:** Fashion-MNIST (28x28 Grayscale Images)

This module implements a custom Convolutional Neural Network (inspired by LeNet) to demonstrate spatial inductive bias and translation equivariance.

* Includes a strict **Ablation Study** comparing Max-Pooling, Average-Pooling, and 1x1 Convolutions.
* Compares the CNN's performance against a standard flattened MLP.
* **Feature Extraction:** The script automatically generates a visual representation of the network's internal activations.

**Learned Feature Maps (First Convolutional Layer):** 

<img width="786" height="184" alt="image" src="https://github.com/user-attachments/assets/6ce3c915-ba6f-4e8d-9dc6-e323546efbf3" />
<img width="784" height="138" alt="image" src="https://github.com/user-attachments/assets/80e42759-f2ee-48fc-9469-e04fc00caf2f" />


---

###  Part III: Natural Language Processing (Seq2Seq GRU)

**File:** `part3_rnn_nlp.py` | **Dataset:** Tatoeba (French-to-English Translation)

This module tackles sequential data of variable lengths using an Encoder-Decoder architecture.

* Replaces standard RNNs with **GRU cells** to mitigate the vanishing gradient problem.
* Implements critical training techniques: **Gradient Clipping** (`max_norm=1.0`) and **Teacher Forcing** (50% ratio).
* Evaluates translation quality using **Greedy Decoding** and the **BLEU-1 Score**.

<img width="838" height="628" alt="image" src="https://github.com/user-attachments/assets/5034a3c1-2dab-4e74-8527-ff4f17fb973b" />


---

##  Project Structure

```text
├── data/                       # (Auto-generated) Datasets download here
├── saved_models/               # (Auto-generated) Trained .pth weights
├── main.py                     # Master CLI Dashboard
├── part1_mlp_tabular.py        # Tabular data modeling
├── part2_cnn_vision.py         # Image classification
├── part3_rnn_nlp.py            # Machine translation
├── feature_maps_visualization.png # CNN Filter Visualization
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

```
