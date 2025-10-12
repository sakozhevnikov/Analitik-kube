import pytest
import pandas as pd
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from modules.data_loader import load_data

def test_load_data():
    """Тест функции загрузки данных."""
    # Пока просто проверяем что функция импортируется
    assert callable(load_data)
    print("✅ Функция load_data импортируется")