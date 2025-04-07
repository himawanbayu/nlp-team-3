# Five BERTs Walk into a Test: Evaluating Science Question Distractors with Language Models

## Installation
- It is recommended to create a virtual environment by running: 
```
python -m venv env
```
- Then, install the required dependencies with:
```
pip install -r requirements.txt
```
## Datasets
We use the test set of the [SciQ dataset](https://huggingface.co/datasets/allenai/sciq) to generate machine-generated distractors.

## Distractor generation
To generate distractors using Qwen models, simply run:
```
python distractors/Qwen_distractors.py
```
The results will be saved in the generated_distractors folder.

## Fine tuning
To fine-tune the models, navigate to the `fine-tuning` folder and run either:
```
python main.py
```
or, if you're using Habrok, run:
```
sbatch jobscript.sh
```
We also provide checkpoint files for BERT models fine-tuned on the SciQ training set:
<table>
  <tr>
    <th>models</th>
    <th>accuracy</th>
    <th>download</th>
  </tr>
  <tr>
    <td>Albert-base-v2</td>
    <td>61.4%</td>
    <td><a href="https://drive.google.com/file/d/1S-tvIRsjhp7wPl5MFb5-2HCa5Tgy_rmW/view?usp=sharing">checkpoint</a></td>
  </tr>
  <tr>
    <td>BERT-base</td>
    <td>65%</td>
    <td><a href="https://drive.google.com/file/d/1vU-bgRTiGsw7hMam7R7AUWIVgP_gVVsf/view?usp=sharing">checkpoint</a></td>
  </tr>
  <tr>
    <td>BERT-large</td>
    <td>65.8%</td>
    <td><a href="https://drive.google.com/file/d/1KAoESi2w8-N7NVhCu6Nt8fHojeiz6r6i/view?usp=sharing">checkpoint</a></td>
  </tr>
  <tr>
    <td>Distilbert-base</td>
    <td>65.4%</td>
    <td><a href="https://drive.google.com/file/d/1Pc6pXSYMz2lqImr5zfJ8sjYKK1u6V5LQ/view?usp=sharing">checkpoint</a></td>
  </tr>
  <tr>
    <td>RoBERTa-base</td>
    <td>57.6%</td>
    <td><a href="https://drive.google.com/file/d/1oP4Jr2CvA8kCvN6hXVgyc_4L5kQlu_J9/view?usp=sharing">checkpoint</a></td>
  </tr>
</table>

## Test Takers test

To evaluate the test-taker models, run:
```
python results/fine-tuned.py
```
You can modify the model by changing the `model_name` and `checkpoint_path` variables inside `fine-tuned.py`.

## Evaluation Metrics
All evaluation metrics are computed in the Jupyter notebook located at:
```
results/qwen/Support_False/Exploration.ipynb
```
This notebook loads the test results for each model and performs the quantitative analysis. It generates the plots and statistics used in the report. All relevant outputs have been preserved.
