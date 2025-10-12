import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception:
        return None  # Только логика, без интерфейса

def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Возвращает информацию о данных для отображения.
    
    Args:
        df: DataFrame для анализа
        
    Returns:
        Dict с информацией о данных
    """
    info = {
        'rows': df.shape[0],
        'columns': df.shape[1], 
        'memory_size': f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB",
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    return info