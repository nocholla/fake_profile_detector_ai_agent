from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df, config, label_encoders=None, scaler=None, tfidf=None):
    """Preprocess profile data"""
    try:
        features = config['features']
        logger.info(f"Using features: {features}")
        df = df[features + ['label']].drop_duplicates().fillna({'age': df['age'].median(), 'country': 'unknown', 'relationshipGoals': 'unknown', 'aboutMe': 'unknown'})
        
        if label_encoders is None:
            label_encoders = {}
            for col in ['country', 'relationshipGoals']:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col])
                logger.info(f"Fitted LabelEncoder for {col} with classes: {label_encoders[col].classes_}")
        else:
            for col in ['country', 'relationshipGoals']:
                try:
                    df[col] = label_encoders[col].transform(df[col])
                except ValueError as e:
                    logger.warning(f"Unknown category in {col}: {e}")
                    df[col] = label_encoders[col].transform(['unknown'] * len(df))
        
        df['subscribed'] = df['subscribed'].astype(int)
        
        if scaler is None:
            scaler = StandardScaler()
            df[['age']] = scaler.fit_transform(df[['age']])
            logger.info("Fitted StandardScaler for age")
        else:
            df[['age']] = scaler.transform(df[['age']])
        
        if tfidf is None:
            tfidf = TfidfVectorizer(max_features=config.get('tfidf_max_features', 100))
            aboutMe_tfidf = tfidf.fit_transform(df['aboutMe'])
            logger.info(f"Fitted TfidfVectorizer with {len(tfidf.get_feature_names_out())} features")
        else:
            aboutMe_tfidf = tfidf.transform(df['aboutMe'])
        
        X_other = df.drop(columns=['label', 'aboutMe'])
        X = hstack([X_other.values, aboutMe_tfidf])
        y = df['label']
        
        logger.info(f"Preprocessed data: X shape={X.shape}, y shape={y.shape}")
        return X, y, label_encoders, scaler, tfidf
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise