# Metternich

Deep learning using BERT embeddings for speech act conflict prediction on the UN General Debates Corpus

## Install

Use Python 3.11.9 or other compatible version. Install the Metternich project:

`git clone git@github.com:michelcapelle/metternich.git`
`cd metternich`

### Dependencies

Install requirements:

`pip install -r requirements.txt`

### UN General Debates Data

Install speech corpus:

`git clone git@github.com:jradius/un-general-debates.git`

## Set-up

Check (and update) settings:

`.env`

### Preprocessing parameters

- DO_TEST: Perform quick test run
- START_YEAR: First year with data in UNGDC
- END_YEAR: Last year with data in UNGDC
- TRAIN_PCT: Percentage of debates used for training
- VAL_PCT: Percentage of debates used for validation
- PRED_PCT: Percentage of debates used for prediction
- WINDOW_YEARS: Number of years to look ahead for war prediction (minus to look back)
- MAX_WINDOW_YEARS: Maximum number of years to look ahead (or back) for war prediction, i.e., used to keep [START_YEAR, END_YEAR] stable when tuning WINDOW_YEARS
- IS_BENCHMARK: 1 to use benchmark text, 0 to use actual speeches

### Deep learning parameters

- BERT_MODEL: Pretrained BERT model for tokenizer
- MAX_LENGTH: Maximum token length for BERT encoding
- BATCH_SIZE: Batch size for model
- LEARNING_RATE: Learning rate for Adam optimizer
- MAX_EPOCHS: Maximum number of training epochs
- PATIENCE: Early stopping patience
- MIN_DELTA: Minimum change to qualify as an improvement for early stopping
- TF_ENABLE_ONEDNN_OPTS: Disable oneDNN optimizations

### Meta parameters

- HEAD_PRINT: Number of rows to print for data previews
- VERBOSE: Print model details (1)
- CASE_FROM: Print use case using starting from year
- CASE_TO: Print use case using ending with year

## Run

Run single prediction:

`python klemens.py`

### Compare

Compare reproduced results with default .env results:

`data/results.csv`
