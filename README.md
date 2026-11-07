# Audio Fingerprinting Project

This project implements an audio fingerprinting system based on Avery Wang's algorithm for Shazam. 

Both reference and query audio files are in 44.1kHz
# Instructions:

For this project, you'll need a python version > 3.12.
You can check your version by running
for macOS and linux:
``` bash
python --version
```
or
``` bash
python3 --version
```

for Windows:
``` bash
python -V
```

please follow [this guide](https://www.pythoncentral.io/how-to-update-python/) if you need to install python or update your python version.

We don't want to mess with your favourite personal python set up, so we'll create a local environment and activate it. 
If you have multiple python versions installed, please specify the python version which to install in the environment by replacing `python` in the command below with e.g. `python3.12`

For macOS and Linux:
```bash
 python3 -m venv venv
 source venv/bin/activate
 ```

On Windows:

```bash
python3 -m venv workshopenv
workshopenv\Scripts\activate.bat
```

2. Install dependencies:

```bash 
pip install -r requirements.txt
```
If you have issues installing the modules lined out in the requirements, try to update pip
```bash
python3 -m pip install --upgrade pip
```

3. Run the main script for end to end execution using the following command in the terminal:

```bash
python3 main.py
```
This will run the matching query on all files in `audio_data/query` using the files in `audio_data/database` as references.
If `audio_fingerprint_data/fingerprint.pkl` does not exist, the reference fingerprints will be generated.

It will output a ranking of matching reference files and scoring. 
```
Query song: 011298_query.mp3
Match:
  1. 011298.mp3  (score: 680)
  2. 057938.mp3  (score: 3)
  3. 013666.mp3  (score: 2)
  4. 064006.mp3  (score: 2)
  5. 071244.mp3  (score: 2)
```

## Importing files (Input Files)

The system takes .mp3 files as input for fingerprinting. Import the database recordings to the database_recordings directory.
```
'./audio_data/database'
```
Import query files 
```
'./audio_data/query'
```

## Running the workshop notebook
With your virtual environment activated, run on Mac:
```bash
jupyter notebook notebooks/workshop.ipynb
```
on Windows:
```bash
jupyter notebook notebooks\workshop.ipynb
```

This will open a browser page on your local host. The address is printed to the console output, e.g. `http://localhost:8888/`

References: <br>
[1] Wang, A. (2003). An industrial strength audio search algorithm. In Proceedings of the 4th International Conference on Music Information Retrieval (ISMIR 2003) (pp. 7–13) <br>
[2] Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A Dataset For Music Analysis [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5HW28.


