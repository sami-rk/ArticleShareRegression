"""
Configuration constants for Task 2: Regression Pipeline
"""

RANDOM_STATE = 42
N_FOLDS = 5
ALPHA = 0.5

TRAIN_DATA_PATH = 'datasets/train.xlsx'
TEST_DATA_PATH = 'datasets/test.xlsx'

IMPUTE_COLS = ['video_count', 'avg_word_length', 'keyword_count', 'title_subjectivity']

CHANNEL_ORD = {
    'world': 0, 'tech': 1, 'entertainment': 2, 'business': 3,
    'unknown': 4, 'social_media': 5, 'lifestyle': 6
}
