import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_data(config):
    """Load and merge CSV files"""
    try:
        data_dir = Path(config['data_dir'])
        # Validate data directory
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at {data_dir.resolve()}")
        
        profiles_path = data_dir / 'Profiles.csv'
        if not profiles_path.exists():
            raise FileNotFoundError(f"Profiles.csv not found at {profiles_path.resolve()}")
        
        logger.info(f"Loading profiles from {profiles_path.resolve()}")
        profiles = pd.read_csv(profiles_path)
        
        blocked = set(pd.read_csv(data_dir / 'BlockedUsers.csv')['userId']) if (data_dir / 'BlockedUsers.csv').exists() else set()
        declined = set(pd.read_csv(data_dir / 'DeclinedUsers.csv')['userId']) if (data_dir / 'DeclinedUsers.csv').exists() else set()
        deleted = set(pd.read_csv(data_dir / 'DeletedUsers.csv')['userId']) if (data_dir / 'DeletedUsers.csv').exists() else set()
        reported = set(pd.read_csv(data_dir / 'ReportedUsers.csv')['userId']) if (data_dir / 'ReportedUsers.csv').exists() else set()
        
        profiles['label'] = profiles['userId'].apply(lambda x: 1 if x in (blocked | declined | deleted | reported) else 0)
        logger.info(f"Loaded {len(profiles)} profiles with labels: {profiles['label'].value_counts().to_dict()}")
        return profiles
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise