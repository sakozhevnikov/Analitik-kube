"""Базовые тесты для проверки окружения."""
import sys
import os

def test_python_path():
    """Тест что Python видит корень проекта."""
    # Добавляем корень проекта в Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    # Проверяем что путь добавился
    assert project_root in sys.path

def test_modules_import():
    """Тест импорта модулей."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    try:
        # Пробуем импортировать
        from modules.data_loader import get_data_info
        assert callable(get_data_info)
        print("✅ get_data_info imported successfully")
    except ImportError as e:
        assert False, f"Ошибка импорта: {e}"

def test_pandas_available():
    """Тест что pandas доступен."""
    try:
        import pandas as pd
        df = pd.DataFrame({'test': [1, 2, 3]})
        assert len(df) == 3
        print("✅ Pandas works correctly")
    except ImportError:
        assert False, "Pandas не установлен"

def test_data_loader_functionality():
    """Тест функциональности data_loader."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from modules.data_loader import get_data_info
    import pandas as pd
    
    # Создаем тестовые данные
    test_data = pd.DataFrame({
        'name': ['Alice', 'Bob'],
        'age': [25, 30],
        'salary': [50000, 60000]
    })
    
    # Тестируем функцию
    result = get_data_info(test_data)
    
    # Проверяем результат
    assert result['rows'] == 2
    assert result['columns'] == 3
    assert 'age' in result['numeric_columns']
    assert 'name' in result['categorical_columns']
    print("✅ get_data_info works correctly")