from dotenv import load_dotenv
import os
import tensorflow as tf

class KlemensSettings:

    def __init__(self):
        load_dotenv()
        self.bert_model = os.getenv("BERT_MODEL")
        self.max_len = int(os.getenv("MAX_LENGTH"))
        self.is_benchmark = os.getenv("IS_BENCHMARK") == "1"
        self.batch_size = (int)(os.getenv("BATCH_SIZE"))
        self.loss_function = os.getenv("LOSS_FUNCTION")
        self.metric = os.getenv("METRIC")
        self.patience = (int)(os.getenv("PATIENCE"))
        self.min_delta = (float)(os.getenv("MIN_DELTA"))
        self.max_epochs = (int)(os.getenv("MAX_EPOCHS"))
        self.train_pct = (float)(os.getenv("TRAIN_PCT"))
        self.val_pct = (float)(os.getenv("VAL_PCT"))
        self.pred_pct = (float)(os.getenv("PRED_PCT"))
        self.head_print = (int)(os.getenv("HEAD_PRINT"))
        self.start_year = (int)(os.getenv("START_YEAR"))
        self.end_year = (int)(os.getenv("END_YEAR"))
        self.max_window_years = (int)(os.getenv("MAX_WINDOW_YEARS"))
        self.window_years = (int)(os.getenv("WINDOW_YEARS"))
        self.learning_rate = (float)(os.getenv("LEARNING_RATE"))
        self.verbose = (int)(os.getenv("VERBOSE"))
        self.case_from = (int)(os.getenv("CASE_FROM"))
        self.case_to = (int)(os.getenv("CASE_TO"))
        self.random_state = 42
        self.datatype = self.get_datatype()
        self.drops_conflicts = [
            'conflict_id', 
            'location', 
            'side_a',
            'side_a_id', 
            'side_a_2nd', 
            'side_b', 
            'side_b_id', 
            'side_b_2nd', 
            'incompatibility', 
            'territory_name', 
            'cumulative_intensity', 
            'start_date', 
            'start_prec', 
            'start_date2', 
            'start_prec2', 
            'ep_end', 
            'ep_end_date', 
            'ep_end_prec', 
            'version', 
            'type_of_conflict', 
            'region', 
            'intensity_level',
        ]
        self.do_test = os.getenv("DO_TEST") == '1'
        if self.do_test:
            self.start_year = 1946
            self.end_year = self.start_year + 2
            self.window_years = 1
            self.max_window_years = self.window_years
    
    @staticmethod
    def get_datatype():
        return tf.float32
