This folder contains the test and results made by Hongxu Zhou. 

I tested the performance of `bert_base_uncased` on the tests with llama-generated and SciQ original distractors respectively. 

Regarding the format of input, I further tested two varieties: 
1. **query-first**: namely *question_context* in scrips and results
2. **bert format**: the official format used for BERT training. 

I also tested the impact of context on model performance based on the scripts of `bert format`. The scripts are titled 'no_context'. 

Their results in CSV files follow the same format. 

To avoid any ambiguity or misunderstanding:
*`llama-generated distractors` correspond to the scripts/results with 'generated/generalised' in their file names 
*`SciQ original distractors` correspond to the ones with 'original' in their filenames

All scripts are tested with duplicated choices kept, and choices randomly shuffled.
Accuracy is calculated by directly comparing the correct answer with the model's choice.
