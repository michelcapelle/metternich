import importlib.util
import os
import tensorflow as tf
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import functions
import time
import tensorflow_hub as hub

print(f"\n[1/12] UNGDC data") # https://academic.oup.com/isq/article-abstract/68/1/sqae001/7587491?redirectedFrom=fulltext
spec = importlib.util.spec_from_file_location("transform_UNGDC", "un-general-debates/data/transform_UNGDC.py") # https://ucdp.uu.se/downloads/index.html#armedconflict
transform_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transform_module)
original_dir = os.getcwd()
os.chdir('un-general-debates/data')
corpus = transform_module.get_data()
is_benchmark = os.getenv("IS_BENCHMARK") == "1"
if is_benchmark:
    corpus['text'] = "To those who can hear me, I say - do not despair. The misery that is now upon us is but the passing of greed - the bitterness of men who fear the way of human progress. The hate of men will pass, and dictators die, and the power they took from the people will return to the people. And so long as men die, liberty will never perish…" # https://www.charliechaplin.com/en/synopsis/articles/29-The-Great-Dictator-s-Speech
os.chdir(original_dir)
print(f"Shape: {corpus.shape}")
head = 5
print(corpus.head(head))
print("By year counts:")
year_counts = corpus.groupby('year').size()
print(year_counts.describe())
corpus['text_length'] = corpus['text'].str.len()
print("By word counts:")
corpus['word_count'] = corpus['text'].str.split().str.len()
print(corpus['word_count'].describe())
print(f"[1/12] Done!")

print(f"\n[2/12] Country code data")
names = ['code', 'abbrev', 'country', 'start_date', 'end_date']
ii_system = pd.read_csv('data/iisystem.dat', sep='\t', header=None, names=names, encoding='latin-1') # http://ksgleditsch.com/data-4.html
drops = ['country', 'start_date', 'end_date']
ii_system = ii_system.drop(drops, axis=1)
print(f"Shape: {ii_system.shape}")
print(ii_system.head(head))
print(f"[2/12] Done!")

print(f"\n[3/12] UCDP conflict data")
ucdp = pd.read_csv('data/UcdpPrioConflict_v25_1.csv')
ucdp = ucdp[ucdp['intensity_level'] == 2]
drops = ['conflict_id', 'location', 'side_a', 'side_a_id', 'side_a_2nd', 'side_b', 'side_b_id', 'side_b_2nd', 'incompatibility', 'territory_name', 'cumulative_intensity', 'start_date', 'start_prec', 'start_date2', 'start_prec2', 'ep_end', 'ep_end_date', 'ep_end_prec', 'version', 'type_of_conflict', 'region', 'intensity_level']
ucdp = ucdp.drop(drops, axis=1)
print(f"Shape: {ucdp.shape}")
headers = ucdp.head(head)
print(headers)
print(f"[3/12] Done!")

print(f"\n[4/12] Split into training and prediction sets")
corpus['year'] = corpus['year'].astype(int)
load_dotenv()
split_year = (int)(os.getenv("SPLIT_YEAR"))
train_set = corpus[corpus['year'] < split_year].copy()
end_year = (int)(os.getenv("END_YEAR"))
max_window_years = 5
pred_set = corpus[(corpus['year'] >= split_year + max_window_years) & (corpus['year'] <= end_year - max_window_years)].copy()
train_len = len(train_set)
print(f"Training set: {train_len} records < {split_year}")
pred_len = len(pred_set)
final_year = end_year - max_window_years
print(f"Prediction set: {pred_len} records >= {split_year} <= {final_year}")
print("Training set by year counts:")
year_counts = train_set.groupby('year').size()
print(year_counts.describe())
train_set['text_length'] = train_set['text'].str.len()
print("Training set by word counts:")
train_set['word_count'] = train_set['text'].str.split().str.len()
print(train_set['word_count'].describe())
print("Prediction set by year counts:")
year_counts = pred_set.groupby('year').size()
print(year_counts.describe())
pred_set['text_length'] = pred_set['text'].str.len()
print("Prediction set by word counts:")
pred_set['word_count'] = pred_set['text'].str.split().str.len()
print(pred_set['word_count'].describe())
print(f"[4/12] Done!")

print(f"\n[5/12] Enoding for training")
bert_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
bert_encoder = hub.KerasLayer(bert_url, trainable=True, name="bert")
train_texts = train_set["text"].tolist()
max_len = 512
train_enc = functions.encode_texts(train_texts, max_len)
train_input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
train_input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
train_input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")
train_bert_inputs = {
    "input_word_ids": train_input_ids,
    "input_mask": train_input_mask,
    "input_type_ids": train_input_type_ids,
}
print(f"[5/12] Done!")

print(f"\n[6/12] Enoding for prediction")
pred_texts = pred_set["text"].tolist()
pred_texts_enc = functions.encode_texts(pred_texts, max_len)
pred_input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
pred_input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
pred_input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")
pred_bert_inputs = {
    "input_word_ids": pred_input_ids,
    "input_mask": pred_input_mask,
    "input_type_ids": pred_input_type_ids,
}
print(f"[6/12] Done!")

print(f"\n[7/12] Labels for training")
window_years = (int)(os.getenv("WINDOW_YEARS"))
print(f"Window years: {window_years}")
y_train = functions.get_labels(train_set, ucdp, window_years, ii_system)
mean = np.mean(y_train)
print(f"War probability: {mean:.4f}")
print(f"[7/12] Done!")

print(f"\n[8/12] Labels for prediction")
y_true = functions.get_labels(pred_set, ucdp, window_years, ii_system)
mean = np.mean(y_true)
print(f"War probability: {mean:.4f}")
print(f"[8/12] Done!")

print(f"\n[9/12] Compile")
bert_outputs = bert_encoder(train_bert_inputs)
cls_token = bert_outputs["pooled_output"]
x = tf.keras.layers.Dropout(0.1)(cls_token)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=train_bert_inputs, outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()
print(f"[9/12] Done!")

print(f"\n[10/12] Fit")
verbose = 1
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=verbose, mode='min', min_delta=0.0001)
start_time = time.time()
history = model.fit(train_enc, y_train, validation_split=0.2, epochs=10, verbose=verbose, callbacks=[early_stopping], batch_size=2)
end_time = time.time()
training_time = end_time - start_time
print(f"[10/12] Done!")

print(f"\n[11/12] Predict")
y_prob = model.predict(pred_texts_enc, verbose=verbose)
y_pred = (y_prob > 0.5).astype(int).flatten()
print(f"[11/12] Done!")

print(f"\n[12/12] Print")
pred_len = len(pred_set)
sample_indices = np.random.choice(pred_len, 25, replace=False)
for idx in sample_indices:
    true_label = y_true[idx]
    prob = y_prob[idx][0]
    country = pred_set['country'].values[idx]
    year = pred_set['year'].values[idx]
    print(f"Sample {idx}: Country={country}, Year={year}, True={true_label}, Probability={prob:.4f}")
loss, accuracy = model.evaluate(pred_texts_enc, y_true, verbose=verbose)
print(f"Split year: {split_year}")
print(f"End year: {end_year}")
print(f"Window years: {window_years}")
print(f"Is benchmark: {is_benchmark}")
print(f"Training time: {training_time:.0f} seconds")
print(f"Predict accuracy: {accuracy:.5f}")
print(f"[12/12] Done!")

print(f"[*] Completed!")
