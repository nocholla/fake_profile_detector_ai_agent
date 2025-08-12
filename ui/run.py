import pandas as pd
import numpy as np
import yaml
import joblib
import logging
import argparse
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pyod.models.iforest import IForest
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import faiss
from scipy.sparse import hstack
from sklearn.metrics import classification_report
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = Path('config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Define paths
DATA_DIR = Path(config['data_dir'])
MODEL_DIR = Path(config['model_dir'])
CSV_FILES = {
    'profiles': DATA_DIR / 'Profiles.csv',
    'blocked': DATA_DIR / 'BlockedUsers.csv',
    'declined': DATA_DIR / 'DeclinedUsers.csv',
    'deleted': DATA_DIR / 'DeletedUsers.csv',
    'reported': DATA_DIR / 'ReportedUsers.csv'
}
MODEL_PATHS = {
    'rf_model': MODEL_DIR / 'rf_model.pkl',
    'xgb_model': MODEL_DIR / 'xgb_model.pkl',
    'iforest_model': MODEL_DIR / 'iforest_model.pkl',
    'anomaly_net': MODEL_DIR / 'anomaly_net.pth',
    'tfidf': MODEL_DIR / 'tfidf_vectorizer.pkl',
    'label_encoders': MODEL_DIR / 'label_encoders.pkl',
    'scaler': MODEL_DIR / 'scaler.pkl',
    'faiss_index': MODEL_DIR / 'faiss_index.index'
}

# Neural Network for Anomaly Detection
class AnomalyNet(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

@st.cache_resource
def load_data():
    """Load and merge CSV files"""
    try:
        profiles = pd.read_csv(CSV_FILES['profiles'])
        blocked = set(pd.read_csv(CSV_FILES['blocked'])['userId']) if CSV_FILES['blocked'].exists() else set()
        declined = set(pd.read_csv(CSV_FILES['declined'])['userId']) if CSV_FILES['declined'].exists() else set()
        deleted = set(pd.read_csv(CSV_FILES['deleted'])['userId']) if CSV_FILES['deleted'].exists() else set()
        reported = set(pd.read_csv(CSV_FILES['reported'])['userId']) if CSV_FILES['reported'].exists() else set()
        
        profiles['label'] = profiles['userId'].apply(lambda x: 1 if x in (blocked | declined | deleted | reported) else 0)
        return profiles
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

@st.cache_resource
def preprocess_data(df, label_encoders=None, scaler=None, tfidf=None):
    """Preprocess profile data"""
    features = ['age', 'country', 'subscribed', 'relationshipGoals', 'aboutMe']
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
        tfidf = TfidfVectorizer(max_features=100)
        aboutMe_tfidf = tfidf.fit_transform(df['aboutMe'])
    else:
        aboutMe_tfidf = tfidf.transform(df['aboutMe'])
    
    X_other = df.drop(columns=['label', 'aboutMe'])
    X = hstack([X_other.values, aboutMe_tfidf])
    y = df['label']
    
    return X, y, label_encoders, scaler, tfidf

@st.cache_resource
def build_rag_index(df):
    """Build FAISS index for RAG"""
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = (df['aboutMe'].astype(str) + ' ' + df['relationshipGoals'].astype(str)).tolist()
        embeddings = embedding_model.encode(texts)
        
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings.astype(np.float32))
        return faiss_index, embedding_model
    except Exception as e:
        logger.error(f"Error building RAG index: {e}")
        raise

@st.cache_resource
def train_anomaly_net(X, input_dim):
    """Train PyTorch autoencoder for anomaly detection"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        anomaly_net = AnomalyNet(input_dim).to(device)
        optimizer = torch.optim.Adam(anomaly_net.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X.toarray()).to(device)
        
        for epoch in range(100):
            anomaly_net.train()
            optimizer.zero_grad()
            output = anomaly_net(X_tensor)
            loss = criterion(output, X_tensor)
            loss.backward()
            optimizer.step()
        
        return anomaly_net
    except Exception as e:
        logger.error(f"Error training anomaly network: {e}")
        raise

class FakeProfileDetector:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.iforest = IForest(contamination=0.1, random_state=42)
        self.anomaly_net = None
        self.label_encoders = None
        self.scaler = None
        self.tfidf = None
        self.faiss_index = None
        self.embedding_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self):
        """Train the fake profile detector"""
        try:
            df = load_data()
            X, y, self.label_encoders, self.scaler, self.tfidf = preprocess_data(df)
            self.faiss_index, self.embedding_model = build_rag_index(df)
            
            # Train models
            self.rf_model.fit(X, y)
            self.xgb_model.fit(X, y)
            self.iforest.fit(X)
            self.anomaly_net = train_anomaly_net(X, X.shape[1])
            
            # Save models and encoders
            MODEL_DIR.mkdir(exist_ok=True)
            joblib.dump(self.rf_model, MODEL_PATHS['rf_model'])
            joblib.dump(self.xgb_model, MODEL_PATHS['xgb_model'])
            joblib.dump(self.iforest, MODEL_PATHS['iforest_model'])
            torch.save(self.anomaly_net.state_dict(), MODEL_PATHS['anomaly_net'])
            joblib.dump(self.tfidf, MODEL_PATHS['tfidf'])
            joblib.dump(self.label_encoders, MODEL_PATHS['label_encoders'])
            joblib.dump(self.scaler, MODEL_PATHS['scaler'])
            faiss.write_index(self.faiss_index, str(MODEL_PATHS['faiss_index']))
            
            # Evaluate
            y_pred = self.rf_model.predict(X)
            logger.info("Random Forest Results:\n%s", classification_report(y, y_pred, target_names=["Real", "Fake"]))
            y_pred = self.xgb_model.predict(X)
            logger.info("XGBoost Results:\n%s", classification_report(y, y_pred, target_names=["Real", "Fake"]))
            y_pred = self.iforest.predict(X)
            logger.info("Isolation Forest Results:\n%s", classification_report(y, y_pred, target_names=["Real", "Fake"]))
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def load_models(self):
        """Load pre-trained models and encoders"""
        try:
            self.rf_model = joblib.load(MODEL_PATHS['rf_model'])
            self.xgb_model = joblib.load(MODEL_PATHS['xgb_model'])
            self.iforest = joblib.load(MODEL_PATHS['iforest_model'])
            self.anomaly_net = AnomalyNet(X.shape[1]).to(self.device)
            self.anomaly_net.load_state_dict(torch.load(MODEL_PATHS['anomaly_net']))
            self.tfidf = joblib.load(MODEL_PATHS['tfidf'])
            self.label_encoders = joblib.load(MODEL_PATHS['label_encoders'])
            self.scaler = joblib.load(MODEL_PATHS['scaler'])
            self.faiss_index = faiss.read_index(str(MODEL_PATHS['faiss_index']))
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict(self, profile_dict):
        """Predict if a profile is fake"""
        try:
            temp_df = pd.DataFrame([{
                'age': profile_dict.get('age', 0),
                'country': profile_dict.get('country', 'unknown'),
                'relationshipGoals': profile_dict.get('relationshipGoals', 'unknown'),
                'subscribed': int(profile_dict.get('subscribed', False)),
                'aboutMe': profile_dict.get('aboutMe', 'unknown')
            }])
            
            for col in ['country', 'relationshipGoals']:
                temp_df[col] = self.label_encoders[col].transform(temp_df[col])
            
            temp_df[['age']] = self.scaler.transform(temp_df[['age']])
            about_me_tfidf = self.tfidf.transform(temp_df['aboutMe'])
            final_input = hstack([temp_df.drop(columns=['aboutMe']).values, about_me_tfidf])
            
            # RAG: Find similar profiles
            text = profile_dict.get('aboutMe', 'unknown') + ' ' + profile_dict.get('relationshipGoals', 'unknown')
            query_embedding = self.embedding_model.encode([text])[0]
            _, indices = self.faiss_index.search(np.array([query_embedding]).astype(np.float32), k=5)
            
            # Model predictions
            rf_pred = self.rf_model.predict_proba(final_input)[0][1]
            xgb_pred = self.xgb_model.predict_proba(final_input)[0][1]
            if_pred = self.iforest.decision_function(final_input)[0]
            
            # Anomaly detection
            final_input_tensor = torch.FloatTensor(final_input.toarray()).to(self.device)
            self.anomaly_net.eval()
            with torch.no_grad():
                reconstructed = self.anomaly_net(final_input_tensor)
                anomaly_score = torch.mean((final_input_tensor - reconstructed) ** 2, dim=1).cpu().numpy()[0]
            
            # Combine predictions
            final_score = (0.3 * rf_pred + 0.3 * xgb_pred - 0.2 * if_pred + 0.2 * anomaly_score)
            
            return {
                'is_fake': final_score > 0.5,
                'confidence': float(final_score),
                'similar_profiles': indices[0].tolist(),
                'reasons': self._generate_reasons(rf_pred, xgb_pred, if_pred, anomaly_score)
            }
        except Exception as e:
            logger.error(f"Error predicting profile: {e}")
            raise
    
    def _generate_reasons(self, rf_pred, xgb_pred, if_pred, anomaly_score):
        """Generate reasons for prediction"""
        reasons = []
        if rf_pred > 0.7:
            reasons.append("High Random Forest confidence in fake profile detection")
        if xgb_pred > 0.7:
            reasons.append("High XGBoost confidence in fake profile detection")
        if if_pred < -0.5:
            reasons.append("Isolation Forest indicates significant anomaly")
        if anomaly_score > 0.5:
            reasons.append("Neural network detects unusual profile patterns")
        return reasons or ["Profile characteristics deviate from typical patterns"]

def run_streamlit():
    """Run Streamlit UI"""
    st.title("🚩 Fake Profile Detector")
    st.write("Enter profile details to detect potential fake profiles.")
    
    with st.form("profile_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=21)
        country = st.text_input("Country", value="Ghana")
        about_me = st.text_area("About Me", value="Hi I'm looking for serious love, not games.")
        relationship_goals = st.selectbox("Relationship Goals", ["Long-term", "Short-term", "Marriage", "Casual", "unknown"])
        subscribed = st.checkbox("Subscribed", value=False)
        submitted = st.form_submit_button("Detect Profile")
        
        if submitted:
            profile = {
                'age': age,
                'country': country,
                'aboutMe': about_me,
                'relationshipGoals': relationship_goals,
                'subscribed': subscribed
            }
            detector = FakeProfileDetector()
            try:
                detector.load_models()
                result = detector.predict(profile)
                st.write(f"**Prediction**: {'Fake' if result['is_fake'] else 'Real'}")
                st.write(f"**Confidence**: {result['confidence']:.2f}")
                st.write("**Reasons**:")
                for reason in result['reasons']:
                    st.write(f"- {reason}")
                st.write(f"**Similar Profile Indices**: {result['similar_profiles']}")
                
                # Save result
                output_df = pd.DataFrame([{
                    'user_id': profile.get('user_id', 'unknown'),
                    'prediction': 'Fake' if result['is_fake'] else 'Real',
                    'confidence': result['confidence'],
                    'reasons': '; '.join(result['reasons'])
                }])
                output_df.to_csv(DATA_DIR / 'fake_profile_predictions.csv', mode='a', index=False)
            except Exception as e:
                st.error(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fake Profile Detector")
    parser.add_argument('--mode', choices=['train', 'predict', 'streamlit'], default='predict')
    parser.add_argument('--user_id', type=str, default='unknown')
    parser.add_argument('--age', type=int, default=21)
    parser.add_argument('--country', type=str, default='unknown')
    parser.add_argument('--about_me', type=str, default='unknown')
    parser.add_argument('--relationship_goals', type=str, default='unknown')
    parser.add_argument('--subscribed', action='store_true')
    
    args = parser.parse_args()
    
    if args.mode == 'streamlit':
        run_streamlit()
    else:
        detector = FakeProfileDetector()
        if args.mode == 'train':
            detector.train()
        elif args.mode == 'predict':
            profile = {
                'user_id': args.user_id,
                'age': args.age,
                'country': args.country,
                'aboutMe': args.about_me,
                'relationshipGoals': args.relationship_goals,
                'subscribed': args.subscribed
            }
            try:
                detector.load_models()
                result = detector.predict(profile)
                print(f"Prediction: {'Fake' if result['is_fake'] else 'Real'}")
                print(f"Confidence: {result['confidence']:.2f}")
                print("Reasons:")
                for reason in result['reasons']:
                    print(f"- {reason}")
                print(f"Similar Profile Indices: {result['similar_profiles']}")
                
                # Save result
                output_df = pd.DataFrame([{
                    'user_id': args.user_id,
                    'prediction': 'Fake' if result['is_fake'] else 'Real',
                    'confidence': result['confidence'],
                    'reasons': '; '.join(result['reasons'])
                }])
                output_df.to_csv(DATA_DIR / 'fake_profile_predictions.csv', mode='a', index=False)
            except Exception as e:
                logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()