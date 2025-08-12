import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_data(config):
    """Load and merge CSV files"""
    data_dir = Path(config['data_dir'])
    profiles = pd.read_csv(data_dir / 'Profiles.csv')
    blocked = set(pd.read_csv(data_dir / 'BlockedUsers.csv')['userId']) if (data_dir / 'BlockedUsers.csv').exists() else set()
    declined = set(pd.read_csv(data_dir / 'DeclinedUsers.csv')['userId']) if (data_dir / 'DeclinedUsers.csv').exists() else set()
    deleted = set(pd.read_csv(data_dir / 'DeletedUsers.csv')['userId']) if (data_dir / 'DeletedUsers.csv').exists() else set()
    reported = set(pd.read_csv(data_dir / 'ReportedUsers.csv')['userId']) if (data_dir / 'ReportedUsers.csv').exists() else set()
    
    profiles['label'] = profiles['userId'].apply(lambda x: 1 if x in (blocked | declined | deleted | reported) else 0)
    return profiles