"""
Feature engineering for Task 2: Regression Pipeline
"""

import numpy as np
from config import CHANNEL_ORD


def engineer_features(df, median_fills):
    """Engineer features for the regression model."""
    from preprocessing import apply_imputation

    df = apply_imputation(df, median_fills)

    df['channel_enc'] = df['channel'].map(CHANNEL_ORD).fillna(4)

    df['total_links'] = df['link_count'] + df['internal_link_count']
    df['media_count'] = df['image_count'] + df['video_count']
    df['has_video'] = (df['video_count'] > 0).astype(int)
    df['links_per_word'] = df['link_count'] / (df['word_count'] + 1)
    df['images_per_word'] = df['image_count'] / (df['word_count'] + 1)

    df['sentiment_ratio'] = df['positive_word_rate'] / (df['negative_word_rate'] + 1e-6)
    df['pos_neg_diff'] = df['positive_word_rate'] - df['negative_word_rate']
    df['abs_polarity'] = df['content_polarity'].abs()
    df['abs_title_polarity'] = df['title_polarity'].abs()
    df['polar_subj_product'] = df['title_polarity'] * df['title_subjectivity']
    df['polarity_x_subj'] = df['content_polarity'] * df['content_subjectivity']

    topic_cols = ['topic_0_score', 'topic_1_score', 'topic_2_score', 'topic_3_score']
    df['dominant_topic'] = df[topic_cols].idxmax(axis=1).map(
        {'topic_0_score': 0, 'topic_1_score': 1,
         'topic_2_score': 2, 'topic_3_score': 3})
    df['max_topic_score'] = df[topic_cols].max(axis=1)
    df['topic_concentration'] = df['max_topic_score'] - df[topic_cols].min(axis=1)
    topic_arr = df[topic_cols].values + 1e-9
    df['topic_entropy'] = -(topic_arr * np.log(topic_arr)).sum(axis=1)

    df['word_keyword_ratio'] = df['word_count'] / (df['keyword_count'] + 1)
    df['log_word_count'] = np.log1p(df['word_count'])
    df['log_link_count'] = np.log1p(df['link_count'])
    df['log_image_count'] = np.log1p(df['image_count'])
    df['log_keyword_count'] = np.log1p(df['keyword_count'])

    df['is_weekend_int'] = df['is_weekend'].astype(int)
    df['weekday_sin'] = np.sin(2 * np.pi * df['publish_weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['publish_weekday'] / 7)

    df['weekend_x_channel'] = df['is_weekend_int'] * df['channel_enc']

    return df
