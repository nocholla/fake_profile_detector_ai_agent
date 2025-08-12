from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def preprocess_data(df, config, label_encoders=None, scaler=None, tfidf=None):
    """Preprocess profile data"""
    features = config['features']
    df = df[features + ['label']].drop_duplicates().fillna({'age': df['age'].median(), 'country': 'unknown', 'relationshipGoals': 'unknown', 'aboutMe': 'unknown'})
    
    if label_encoders is None:
        label_encoders = {}
        for col in ['country', 'relationshipGoals']:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])
    else:
        for col in ['country', 'relationshipGoals']:
            df[col] = label_encoders[col].transform(df[col])
    
    df['subscribed'] = df['subscribed'].astype(int)
    
    if scaler is None:
        scaler = StandardScaler()
        df[['age']] = scaler.fit_transform(df[['age']])
    else:
        df[['age']] = scaler.transform(df[['age']])
    
    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=config['tfidf_max_features'])
        aboutMe_tfidf = tfidf.fit_transform(df['aboutMe'])
    else:
        aboutMe_tfidf = tfidf.transform(df['aboutMe'])
    
    X_other = df.drop(columns=['label', 'aboutMe'])
    X = hstack([X_other.values, aboutMe_tfidf])
    y = df['label']
    
    return X, y, label_encoders, scaler, tfidf