import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / 'config.yaml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

class AnomalyNet(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_anomaly_net(X, input_dim):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        anomaly_net = AnomalyNet(input_dim).to(device)
        optimizer = torch.optim.Adam(anomaly_net.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        logger.info(f"Converting X to dense tensor, shape={X.shape}")
        X_tensor = torch.FloatTensor(X.toarray()).to(device)
        logger.info(f"X_tensor created, shape={X_tensor.shape}")
        
        # Handle small datasets
        if X_tensor.shape[0] < 10:
            logger.warning(f"Small dataset detected ({X_tensor.shape[0]} samples). Training may be unstable.")
        
        epochs = config.get('anomaly_epochs', 100)
        logger.info(f"Training for {epochs} epochs")
        for epoch in range(epochs):
            anomaly_net.train()
            optimizer.zero_grad()
            try:
                output = anomaly_net(X_tensor)
                loss = criterion(output, X_tensor)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Invalid loss at epoch {epoch}: {loss.item()}")
                    raise ValueError("Training stopped due to invalid loss")
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            except Exception as e:
                logger.error(f"Error in training loop at epoch {epoch}: {e}")
                raise
        
        logger.info("Finished training AnomalyNet")
        return anomaly_net
    except Exception as e:
        logger.error(f"Error training anomaly network: {e}")
        raise