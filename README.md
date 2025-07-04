ASL Sign Language Letter Recognition

 A machine learning project to recognize American Sign Language (ASL) letters (A-Y, excluding J and Z) using a CNN model trained on the Sign Language MNIST dataset.
 ## Features
 - CNN model with 94.53% test accuracy.
 - Data preprocessing with label filtering (24 classes).
 - MLOps with Git and DVC for model versioning.
 - Jupyter Notebook (`train_model.ipynb`) for training and visualization.
 - Python script (`train_model.py`) for automation.
 ## Setup
 1. Clone the repo:
    ```bash
    git clone https://github.com/Akhila4812/asl-letter-recognition.git
    cd asl-letter-recognition
    ```
 2. Create and activate virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate.ps1  # Windows
    ```
 3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
 4. Download data (Sign Language MNIST) and place in `data/`:
    - `sign_mnist_train.csv`
    - `sign_mnist_test.csv`
 5. Run the notebook:
    ```bash
    jupyter notebook train_model.ipynb
    ```
 ## MLOps
 - **Git**: Version control for code (`train_model.py`, `train_model.ipynb`).
 - **DVC**: Tracks `asl_model.h5` (2.76 MB).
 - Future: Flask UI, Docker, CI-CD.
 ## Results
 - Test accuracy: 94.53%
 - Visualizations: Sample images and training history plots.
 ## Requirements
 See `requirements.txt` for dependencies, including:
 - TensorFlow 2.17.0
 - Matplotlib 3.9.2
 - Jupyter 1.1.1
 - DVC 3.60.0