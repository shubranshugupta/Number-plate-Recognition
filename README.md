# Number-plate-Recognition
Website to read number plate of car, bike etc.

[comment]: <> (![alt-text]&#40;Client_static/image/My Video.gif&#41;)

## How to Install

> Note: You should have conda install

1. Clone the Repository

``
git clone https://github.com/shubranshugupta/Number-plate-Recognition.git
``

2. Creating conda environment.

    For CPU:

``
conda env create -f conda-cpu.yml
`` <br>

&ensp;&ensp;&ensp;&ensp; For GPU:

``
conda env create -f conda-gpu.yml
``

3. Activating conda environment

``
conda activate number-plate
``

4. Installing library

``
pip install -r requirements.txt
``

5. Download Weight file `variables.data-00000-of-00001` and `variables.index` from <a href='https://drive.google.com/drive/folders/1-1qgUIMvZ9SD56Y8_TQYiC076ppYuYcv?usp=sharing'>Drive</a>

6. Extract and past in `checkpoints/custom-416/variables` folder

7. Running client.py

``
python client.py
``
# Number-plate-Recognition
