import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import settings

class BertBinaryClassifier():

    def __init__(self, settings):
        self.settings = settings
        self.tokenizer = BertTokenizer.from_pretrained(self.settings.bert_model)
        self.model = TFBertForSequenceClassification.from_pretrained(
            self.settings.bert_model, 
            num_labels=1,
        )
        print(f"TensorFlow: v{tf.__version__}")
        self.f1_metric = tf.keras.metrics.F1Score(
            average='micro', 
            name='f1_score',
            threshold=0.5,
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings.learning_rate),
            loss=self.soft_f1_loss,
            metrics=[self.f1_metric],
        )
        self.model.summary()
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.settings.patience, 
            restore_best_weights=True, 
            verbose=self.settings.verbose, 
            mode='min',
            min_delta=self.settings.min_delta,
        )

    def train(self, train_x, train_y, val_x, val_y):
        train_y_shaped = tf.cast(train_y, self.settings.datatype)
        val_y_shaped = tf.cast(val_y, self.settings.datatype)
        self.model.fit(
            train_x,
            train_y_shaped,
            validation_data=(val_x, val_y_shaped),
            epochs=self.settings.max_epochs,
            verbose=self.settings.verbose, 
            callbacks=[self.early_stopping],
            batch_size=self.settings.batch_size,
        )

    def predict(self, x):
        return self.model.predict(x, verbose=self.settings.verbose)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=self.settings.verbose)

    def encode(self, x):
        enc = self.tokenizer(
            x,
            max_length=self.settings.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    @staticmethod
    def soft_f1_loss(y_true, y_pred, eps=1e-7):
        y_pred = tf.math.sigmoid(y_pred)
        y_true = tf.cast(y_true, settings.KlemensSettings.get_datatype())
        tp = tf.reduce_sum(y_pred * y_true, axis=0)
        fp = tf.reduce_sum(y_pred * (1 - y_true), axis=0)
        fn = tf.reduce_sum((1 - y_pred) * y_true, axis=0)
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        loss = 1 - f1
        return tf.reduce_mean(loss)
