import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import yaml
import joblib
import logging
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pyod.models.iforest import IForest
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from sklearn.metrics import classification_report
import streamlit as st
import functools
import sys
import os
from src.load_data import load_data
from src.preprocess_data import preprocess_data
from src.train_anomaly_net import AnomalyNet, train_anomaly_net

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure src directory is in sys.path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

# Load configuration
CONFIG_PATH = BASE_DIR / 'config.yaml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Define paths
DATA_DIR = BASE_DIR / config['data_dir']
MODEL_DIR = BASE_DIR / config['model_dir']
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
    'embeddings': MODEL_DIR / 'profile_embeddings.pkl'
}

# Validate paths and permissions
def validate_paths():
    """Validate data and model paths"""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR.resolve()}")
    if not CSV_FILES['profiles'].exists():
        raise FileNotFoundError(f"Profiles.csv not found at {CSV_FILES['profiles'].resolve()}")
    MODEL_DIR.mkdir(exist_ok=True)
    # Check write permissions
    if not os.access(MODEL_DIR, os.W_OK):
        raise PermissionError(f"No write permission for model directory: {MODEL_DIR.resolve()}")
    logger.info(f"Data directory: {DATA_DIR.resolve()}")
    logger.info(f"Model directory: {MODEL_DIR.resolve()}")

# Conditional caching based on execution mode
def cache_if_streamlit(func):
    is_streamlit = len(sys.argv) >= 2 and 'streamlit' in sys.argv[0].lower() and sys.argv[1].lower() in ['run', 'r']
    if is_streamlit and 'streamlit' in sys.modules:
        return st.cache_data(func)
    return func

@cache_if_streamlit
def build_rag_index(df):
    """Build embeddings for RAG using cosine similarity"""
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = (df['aboutMe'].astype(str) + ' ' + df['relationshipGoals'].astype(str)).tolist()
        embeddings = embedding_model.encode(texts, num_workers=1)
        logger.info(f"Built RAG index with {len(embeddings)} embeddings")
        return embeddings, embedding_model, df.index.tolist()
    except Exception as e:
        logger.error(f"Error building RAG index: {e}")
        raise

class FakeProfileDetector:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.iforest = IForest(contamination=config.get('contamination', 0.1), random_state=42)
        self.anomaly_net = None
        self.label_encoders = None
        self.scaler = None
        self.tfidf = None
        self.embeddings = None
        self.embedding_model = None
        self.profile_indices = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    def train(self):
        """Train the fake profile detector"""
        try:
            validate_paths()
            df = load_data(config)
            X, y, self.label_encoders, self.scaler, self.tfidf = preprocess_data(df, config)
            self.embeddings, self.embedding_model, self.profile_indices = build_rag_index(df)
            
            # Check class balance
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                logger.warning(f"Only one class found in y: {unique_classes}. Skipping XGBoost training.")
                self.xgb_model = None
            else:
                logger.info("Training XGBoost model")
                self.xgb_model.fit(X, y)
                logger.info("XGBoost model trained successfully")
            
            logger.info("Training Random Forest model")
            self.rf_model.fit(X, y)
            logger.info("Random Forest model trained successfully")
            
            logger.info("Training Isolation Forest model")
            self.iforest.fit(X.toarray())
            logger.info("Isolation Forest model trained successfully")
            
            logger.info("Training AnomalyNet")
            self.anomaly_net = train_anomaly_net(X, X.shape[1])
            logger.info("AnomalyNet trained successfully")
            
            # Save models and encoders
            logger.info("Saving models and encoders")
            MODEL_DIR.mkdir(exist_ok=True)
            try:
                logger.info(f"Saving Random Forest model to {MODEL_PATHS['rf_model']}")
                joblib.dump(self.rf_model, MODEL_PATHS['rf_model'])
                if self.xgb_model is not None:
                    logger.info(f"Saving XGBoost model to {MODEL_PATHS['xgb_model']}")
                    joblib.dump(self.xgb_model, MODEL_PATHS['xgb_model'])
                logger.info(f"Saving Isolation Forest model to {MODEL_PATHS['iforest_model']}")
                joblib.dump(self.iforest, MODEL_PATHS['iforest_model'])
                logger.info(f"Saving AnomalyNet to {MODEL_PATHS['anomaly_net']}")
                torch.save(self.anomaly_net.state_dict(), MODEL_PATHS['anomaly_net'])
                logger.info(f"Saving TF-IDF vectorizer to {MODEL_PATHS['tfidf']}")
                joblib.dump(self.tfidf, MODEL_PATHS['tfidf'])
                logger.info(f"Saving label encoders to {MODEL_PATHS['label_encoders']}")
                joblib.dump(self.label_encoders, MODEL_PATHS['label_encoders'])
                logger.info(f"Saving scaler to {MODEL_PATHS['scaler']}")
                joblib.dump(self.scaler, MODEL_PATHS['scaler'])
                logger.info(f"Saving embeddings to {MODEL_PATHS['embeddings']}")
                joblib.dump(self.embeddings, MODEL_PATHS['embeddings'])
            except Exception as e:
                logger.error(f"Error saving models: {e}")
                raise
            
            # Evaluate
            logger.info("Evaluating models")
            y_pred = self.rf_model.predict(X)
            logger.info("Random Forest Results:\n%s", classification_report(y, y_pred, target_names=["Real", "Fake"], zero_division=0))
            if self.xgb_model is not None:
                y_pred = self.xgb_model.predict(X)
                logger.info("XGBoost Results:\n%s", classification_report(y, y_pred, target_names=["Real", "Fake"], zero_division=0))
            y_pred = self.iforest.predict(X.toarray())
            logger.info("Isolation Forest Results:\n%s", classification_report(y, y_pred, target_names=["Real", "Fake"], zero_division=0))
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def load_models(self):
        """Load pre-trained models and encoders"""
        try:
            validate_paths()
            if not MODEL_PATHS['rf_model'].exists():
                raise FileNotFoundError(f"Random Forest model not found at {MODEL_PATHS['rf_model'].resolve()}. Run training with 'python src/run.py --mode train'.")
            self.rf_model = joblib.load(MODEL_PATHS['rf_model'])
            
            if MODEL_PATHS['xgb_model'].exists():
                self.xgb_model = joblib.load(MODEL_PATHS['xgb_model'])
            else:
                logger.warning("XGBoost model not found. Skipping.")
                self.xgb_model = None
            
            if not MODEL_PATHS['iforest_model'].exists():
                raise FileNotFoundError(f"Isolation Forest model not found at {MODEL_PATHS['iforest_model'].resolve()}")
            self.iforest = joblib.load(MODEL_PATHS['iforest_model'])
            
            if not MODEL_PATHS['tfidf'].exists():
                raise FileNotFoundError(f"TF-IDF vectorizer not found at {MODEL_PATHS['tfidf'].resolve()}")
            self.tfidf = joblib.load(MODEL_PATHS['tfidf'])
            
            input_dim = self.tfidf.get_feature_names_out().shape[0] + 3
            if not MODEL_PATHS['anomaly_net'].exists():
                raise FileNotFoundError(f"AnomalyNet model not found at {MODEL_PATHS['anomaly_net'].resolve()}")
            self.anomaly_net = AnomalyNet(input_dim).to(self.device)
            self.anomaly_net.load_state_dict(torch.load(MODEL_PATHS['anomaly_net']))
            
            if not MODEL_PATHS['label_encoders'].exists():
                raise FileNotFoundError(f"Label encoders not found at {MODEL_PATHS['label_encoders'].resolve()}")
            self.label_encoders = joblib.load(MODEL_PATHS['label_encoders'])
            
            if not MODEL_PATHS['scaler'].exists():
                raise FileNotFoundError(f"Scaler not found at {MODEL_PATHS['scaler'].resolve()}")
            self.scaler = joblib.load(MODEL_PATHS['scaler'])
            
            if not MODEL_PATHS['embeddings'].exists():
                raise FileNotFoundError(f"Embeddings not found at {MODEL_PATHS['embeddings'].resolve()}")
            self.embeddings = joblib.load(MODEL_PATHS['embeddings'])
            
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.profile_indices = list(range(len(self.embeddings)))
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
                try:
                    temp_df[col] = self.label_encoders[col].transform(temp_df[col])
                except ValueError:
                    temp_df[col] = self.label_encoders[col].transform(['unknown'])[0]
            
            temp_df[['age']] = self.scaler.transform(temp_df[['age']])
            about_me_tfidf = self.tfidf.transform(temp_df['aboutMe'])
            final_input = hstack([temp_df.drop(columns=['aboutMe']).values, about_me_tfidf])
            
            text = profile_dict.get('aboutMe', 'unknown') + ' ' + profile_dict.get('relationshipGoals', 'unknown')
            query_embedding = self.embedding_model.encode([text], num_workers=1)[0]
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            top_k_indices = np.argsort(similarities)[-5:][::-1]
            similar_profiles = [self.profile_indices[i] for i in top_k_indices]
            
            rf_pred = self.rf_model.predict_proba(final_input)[0][1]
            xgb_pred = self.xgb_model.predict_proba(final_input)[0][1] if self.xgb_model is not None else 0.0
            if_pred = self.iforest.decision_function(final_input.toarray())[0]
            final_input_tensor = torch.FloatTensor(final_input.toarray()).to(self.device)
            self.anomaly_net.eval()
            with torch.no_grad():
                reconstructed = self.anomaly_net(final_input_tensor)
                anomaly_score = torch.mean((final_input_tensor - reconstructed) ** 2, dim=1).cpu().numpy()[0]
            
            xgb_weight = 0.3 if self.xgb_model is not None else 0.0
            final_score = (0.3 * rf_pred + xgb_weight * xgb_pred - 0.2 * if_pred + 0.2 * anomaly_score) / (0.3 + xgb_weight + 0.2 + 0.2)
            
            return {
                'is_fake': final_score > 0.5,
                'confidence': float(final_score),
                'similar_profiles': similar_profiles,
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
        if self.xgb_model is not None and xgb_pred > 0.7:
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
                
                output_df = pd.DataFrame([{
                    'user_id': profile.get('user_id', 'unknown'),
                    'prediction': 'Fake' if result['is_fake'] else 'Real',
                    'confidence': result['confidence'],
                    'reasons': '; '.join(result['reasons'])
                }])
                output_df.to_csv(DATA_DIR / 'fake_profile_predictions.csv', mode='a', index=False, header=not (DATA_DIR / 'fake_profile_predictions.csv').exists())
            except FileNotFoundError as e:
                st.error(f"Model files are missing. Please run 'python src/run.py --mode train' first: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

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
                
                output_df = pd.DataFrame([{
                    'user_id': args.user_id,
                    'prediction': 'Fake' if result['is_fake'] else 'Real',
                    'confidence': result['confidence'],
                    'reasons': '; '.join(result['reasons'])
                }])
                output_df.to_csv(DATA_DIR / 'fake_profile_predictions.csv', mode='a', index=False, header=not (DATA_DIR / 'fake_profile_predictions.csv').exists())
            except FileNotFoundError as e:
                logger.error(f"Model files are missing. Please run 'python src/run.py --mode train' first: {e}")
                raise
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                raise

if __name__ == "__main__":
    main()