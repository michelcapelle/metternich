import tensorflow as tf
import numpy as np
import time
from settings import KlemensSettings
from model import BertBinaryClassifier
from helper import KlemensHelper
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

settings = KlemensSettings()
fn = KlemensHelper(settings)

print(f"\n[1/14] Debates")
debates = fn.load_debates()
print(f"Shape: {debates.shape}")
print(debates.head(settings.head_print))
print("By year counts:")
year_counts = debates.groupby('year').size()
print(year_counts.describe())
debates['text_length'] = debates['text'].str.len()
print("By word counts:")
debates['word_count'] = debates['text'].str.split().str.len()
print(debates['word_count'].describe())
fn.print_case_debates(debates)
debates['year'] = debates['year'].astype(int)
print(f"[1/14] Done!")

print(f"\n[2/14] Country codes")
countries = fn.load_contries_codes()
print(f"Shape: {countries.shape}")
print(countries.head(settings.head_print))
print(f"[2/14] Done!")

print(f"\n[3/14] Conflicts")
conflicts = fn.load_conflicts()
print(f"Shape: {conflicts.shape}")
print(conflicts.head(settings.head_print))
fn.print_case_conflicts(conflicts)
conflicts = conflicts.drop(settings.drops_conflicts, axis=1)
print(f"[3/14] Done!")

print(f"\n[4/14] Split data")
first_year = settings.start_year + settings.max_window_years
final_year = settings.end_year - settings.max_window_years
print(f"Range: [{first_year}, {final_year}]")
debates = debates[(debates['year'] >= first_year) & (debates['year'] <= final_year)].copy()
train_set, debates = train_test_split(debates, test_size=(1.0 - settings.train_pct), random_state=settings.random_state)
ratio = settings.val_pct / (settings.val_pct + settings.pred_pct)
val_set, pred_set = train_test_split(debates, test_size=(1.0 - ratio), random_state=settings.random_state)
print(f"Training set: {len(train_set)} debates")
print(f"Validation set: {len(val_set)} debates")
print(f"Prediction set: {len(pred_set)} debates")
print(f"[4/14] Done!")

print(f"\n[5/14] Compile")
bert = BertBinaryClassifier(settings=settings)
print(f"[5/14] Done!")

print(f"\n[6/14] Training encodings")
train_debates = train_set["text"].tolist()
train_enc = bert.encode(x=train_debates)
print(f"[6/14] Done!")

print(f"\n[7/14] Training labels")
y_train = fn.get_labels(debates=train_set, conflicts=conflicts, countries=countries)
print(f"[7/14] Done!")

print(f"\n[8/14] Validation encodings")
val_debates = val_set["text"].tolist()
val_enc = bert.encode(x=val_debates)
print(f"[8/14] Done!")

print(f"\n[9/14] Validation labels")
y_val = fn.get_labels(debates=val_set, conflicts=conflicts, countries=countries)
print(f"[9/14] Done!")

print(f"\n[10/14] Prediction encodings")
pred_debates = pred_set["text"].tolist()
pred_enc = bert.encode(x=pred_debates)
print(f"[10/14] Done!")

print(f"\n[11/14] Prediction labels")
y_true = fn.get_labels(pred_set, conflicts, countries)
war_prob_pred = np.mean(y_true)
print(f"[11/14] Done!")

print(f"\n[12/14] Fit")
start_time = time.time()
bert.train(train_enc, y_train, val_enc, y_val)
training_time = time.time() - start_time
print(f"[12/14] Done!")

print(f"\n[13/14] Predict")
output = bert.predict(pred_enc)
y_batched_prob = tf.sigmoid(output.logits).numpy()
y_prob = np.concatenate(y_batched_prob, axis=0)
y_pred = (y_prob > 0.5).astype(int).flatten()
print(confusion_matrix(y_true, y_pred))
print(f"[13/14] Done!")

print(f"\n[14/14] Results")
pred_len = len(pred_set)
sample_indices = np.random.choice(pred_len, min(pred_len, settings.head_print), replace=False)
for idx in sample_indices:
    true_label = y_true[idx]
    prob = y_prob[idx]
    country = pred_set['country'].values[idx]
    year = pred_set['year'].values[idx]
    text = pred_set['text'].values[idx]
    print(f"\nSample {idx}: Country={country}, Year={year}, Text={text[:500]}, True={true_label}, Probability={prob:.4f}")
loss, f1_score = bert.evaluate(pred_enc, y_true)
fn.write_results_to_csv(training_time=training_time, f1_score=f1_score, war_prob_pred=war_prob_pred)
print(f"[14/14] Done!")

print(f"\n[*] Completed!")
print(f"Window (y):         {settings.window_years}")
print(f"Is benchmark:       {settings.is_benchmark}")
print(f"War prediction (%): {(war_prob_pred * 100.0):.1f}")
print(f"Training (s):       {training_time:.0f}")
print(f"F1 score:           {f1_score:.5f}")
