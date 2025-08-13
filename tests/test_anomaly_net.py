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
        device = torch.device('cpu')
        logger.info(f"Using device: {device}")
        anomaly_net = AnomalyNet(input_dim).to(device)
        optimizer = torch.optim.Adam(anomaly_net.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        logger.info(f"Converting X to dense tensor, shape={X.shape}")
        X_tensor = torch.FloatTensor(X.toarray()).to(device)
        logger.info(f"X_tensor created, shape={X_tensor.shape}")
        
        for epoch in range(config.get('anomaly_epochs', 100)):
            anomaly_net.train()
            optimizer.zero_grad()
            output = anomaly_net(X_tensor)
            loss = criterion(output, X_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        logger.info("Finished training anomaly network")
        return anomaly_net
    except Exception as e:
        logger.error(f"Error training anomaly network: {e}")
        raise

if __name__ == "__main__":
    # Create a small synthetic dataset for testing
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    X_sparse = csr_matrix(X)
    train_anomaly_net(X_sparse, input_dim=10)