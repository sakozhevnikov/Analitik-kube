import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Базовые классы
class DataProcessor(ABC):
    """Абстрактный базовый класс для обработки данных"""
    
    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        pass

class MLModel(ABC):
    """Абстрактный базовый класс для ML моделей"""
    
    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_category(self) -> str:
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        pass

# Конкретные реализации обработчиков данных
class RemoveNullRows(DataProcessor):
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return data.dropna()
    
    def get_name(self) -> str:
        return "Удаление нулевых строк"
    
    def get_description(self) -> str:
        return "Удалить строки, содержащие пропущенные значения"

class FillNulls(DataProcessor):
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        column = kwargs.get('column')
        method = kwargs.get('method', 'mean')
        custom_value = kwargs.get('custom_value')
        
        if method == 'mean':
            fill_value = data[column].mean()
        elif method == 'median':
            fill_value = data[column].median()
        elif method == 'zero':
            fill_value = 0
        else:
            fill_value = custom_value
            
        data_filled = data.copy()
        data_filled[column] = data_filled[column].fillna(fill_value)
        return data_filled
    
    def get_name(self) -> str:
        return "Заполнение пропусков"
    
    def get_description(self) -> str:
        return "Заполнить пропущенные значения в числовых столбцах"

class CategoricalConverter(DataProcessor):
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        column = kwargs.get('column')
        dummies = pd.get_dummies(data[column], prefix=column)
        data_encoded = pd.concat([data, dummies], axis=1)
        return data_encoded.drop(columns=[column])
    
    def get_name(self) -> str:
        return "Преобразование категорий"
    
    def get_description(self) -> str:
        return "Преобразовать категориальный столбец в числовой с помощью one-hot encoding"

class AnomalyDetector(DataProcessor):
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        column = kwargs.get('column')
        threshold = kwargs.get('threshold', 3.0)
        method = kwargs.get('method')
        
        col_data = data[column].dropna()
        z_scores = np.abs(stats.zscore(col_data))
        anomalies = col_data[z_scores > threshold]
        
        if method and len(anomalies) > 0:
            if method == 'median':
                fill_value = data[column].median()
            elif method == 'mean':
                fill_value = data[column].mean()
            elif method == 'mode':
                fill_value = data[column].mode()[0] if not data[column].mode().empty else data[column].median()
            
            anomaly_indices = col_data[z_scores > threshold].index
            data_filled = data.copy()
            data_filled.loc[anomaly_indices, column] = fill_value
            return data_filled
        
        return data
    
    def get_name(self) -> str:
        return "Поиск аномалий"
    
    def get_description(self) -> str:
        return "Обнаружение выбросов с использованием Z-оценки"

# Конкретные реализации ML моделей
class KMeansModel(MLModel):
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        X = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        kmeans = KMeans(
            n_clusters=kwargs.get('n_clusters', 3),
            init=kwargs.get('init', 'k-means++'),
            max_iter=kwargs.get('max_iter', 300),
            random_state=kwargs.get('random_state', 42)
        )
        clusters = kmeans.fit_predict(X)
        
        metrics = {
            'silhouette': silhouette_score(X, clusters),
            'calinski': calinski_harabasz_score(X, clusters),
            'davies': davies_bouldin_score(X, clusters),
            'n_clusters': len(np.unique(clusters))
        }
        
        return {
            'model': kmeans,
            'clusters': clusters,
            'metrics': metrics,
            'features': numeric_cols
        }
    
    def get_name(self) -> str:
        return "K-Means"
    
    def get_category(self) -> str:
        return "clustering"
    
    def get_description(self) -> str:
        return "Алгоритм кластеризации K-Means"

class DBSCANModel(MLModel):
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        X = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        dbscan = DBSCAN(
            eps=kwargs.get('eps', 0.5),
            min_samples=kwargs.get('min_samples', 5),
            metric=kwargs.get('metric', 'euclidean')
        )
        clusters = dbscan.fit_predict(X)
        
        # Убираем шумовые точки для расчета метрик
        mask = clusters != -1
        if mask.sum() > 1:
            X_clean = X[mask]
            clusters_clean = clusters[mask]
            metrics = {
                'silhouette': silhouette_score(X_clean, clusters_clean),
                'calinski': calinski_harabasz_score(X_clean, clusters_clean),
                'davies': davies_bouldin_score(X_clean, clusters_clean),
                'n_clusters': len(np.unique(clusters_clean))
            }
        else:
            metrics = {'silhouette': 0, 'calinski': 0, 'davies': 0, 'n_clusters': 0}
        
        return {
            'model': dbscan,
            'clusters': clusters,
            'metrics': metrics,
            'features': numeric_cols
        }
    
    def get_name(self) -> str:
        return "DBSCAN"
    
    def get_category(self) -> str:
        return "clustering"
    
    def get_description(self) -> str:
        return "Алгоритм кластеризации DBSCAN"

class PCAModel(MLModel):
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        from sklearn.decomposition import PCA
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        X = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        pca = PCA(
            n_components=kwargs.get('n_components', 2),
            random_state=kwargs.get('random_state', 42)
        )
        X_pca = pca.fit_transform(X)
        
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        return {
            'model': pca,
            'transformed_data': X_pca,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'total_variance': explained_variance.sum(),
            'features': numeric_cols
        }
    
    def get_name(self) -> str:
        return "PCA"
    
    def get_category(self) -> str:
        return "dimensionality_reduction"
    
    def get_description(self) -> str:
        return "Метод главных компонент (PCA)"

class TSNEModel(MLModel):
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        from sklearn.manifold import TSNE
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        X = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        tsne = TSNE(
            n_components=kwargs.get('n_components', 2),
            perplexity=kwargs.get('perplexity', 30),
            learning_rate=kwargs.get('learning_rate', 200),
            random_state=kwargs.get('random_state', 42)
        )
        X_tsne = tsne.fit_transform(X)
        
        return {
            'model': tsne,
            'transformed_data': X_tsne,
            'features': numeric_cols
        }
    
    def get_name(self) -> str:
        return "t-SNE"
    
    def get_category(self) -> str:
        return "dimensionality_reduction"
    
    def get_description(self) -> str:
        return "Стохастическое вложение соседей с t-распределением"

class RandomForestModel(MLModel):
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        # Заглушка для демонстрации
        return {
            'model': None,
            'metrics': {'accuracy': 0.85, 'f1_score': 0.82}
        }
    
    def get_name(self) -> str:
        return "Случайный лес"
    
    def get_category(self) -> str:
        return "classification"
    
    def get_description(self) -> str:
        return "Алгоритм классификации Случайный лес"

class GradientBoostingModel(MLModel):
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        # Заглушка для демонстрации
        return {
            'model': None,
            'metrics': {'accuracy': 0.87, 'f1_score': 0.84}
        }
    
    def get_name(self) -> str:
        return "Градиентный бустинг"
    
    def get_category(self) -> str:
        return "classification"
    
    def get_description(self) -> str:
        return "Алгоритм классификации Градиентный бустинг"

# Фабрики
class DataProcessorFactory:
    _processors = {
        'remove_nulls': RemoveNullRows, #settings ui
        'fill_nulls': FillNulls,
        'categorical': CategoricalConverter,
        'anomaly_detection': AnomalyDetector
    }
    
    @classmethod
    def create_processor(cls, processor_type: str) -> DataProcessor:
        if processor_type not in cls._processors:
            raise ValueError(f"Unknown processor type: {processor_type}")
        return cls._processors[processor_type]()
    
    @classmethod
    def get_available_processors(cls) -> List[str]:
        return list(cls._processors.keys())
    
    @classmethod
    def register_processor(cls, name: str, processor_class):
        cls._processors[name] = processor_class

class Isomap:
    



class MLModelFactory:
    _models = {
        'kmeans': KMeansModel, SettingKMeansModel
        'dbscan': DBSCANModel, SettingKMeansModel
        'pca': PCAModel,
        'tsne': TSNEModel,
        'random_forest': RandomForestModel,
        'gradient_boosting': GradientBoostingModel
    }
    
    @classmethod
    def create_model(cls, model_type: str) -> MLModel:
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._models[model_type]()
    
    @classmethod
    def get_models_by_category(cls, category: str) -> List[str]:
        return [name for name in cls._models.keys() 
                if cls.create_model(name).get_category() == category]
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class):
        cls._models[name] = model_class

# Менеджеры
class DataManager:
    def __init__(self):
        self.processor_factory = DataProcessorFactory()
    
    def get_processor_ui(self, processor_type: str, data: pd.DataFrame):
        processor, settings = self.processor_factory.create_processor(processor_type)
        
        processor.showUI()

        if processor_type == 'remove_nulls':
            return self._ui_remove_nulls(processor, data)
        elif processor_type == 'fill_nulls':
            return self._ui_fill_nulls(processor, data)
        elif processor_type == 'categorical':
            return self._ui_categorical(processor, data)
        elif processor_type == 'anomaly_detection':
            return self._ui_anomaly_detection(processor, data)
    
    def _ui_remove_nulls(self, processor: DataProcessor, data: pd.DataFrame):
        st.subheader(processor.get_name())
        st.write(processor.get_description())
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Текущее количество строк с пропусками: {data.isnull().any(axis=1).sum()}")
            st.info(f"Всего строк в данных: {data.shape[0]}")
        with col2:
            if st.button("Удалить нулевые строки", use_container_width=True):
                return processor.process(data)
        return data
    
    def _ui_fill_nulls(self, processor: DataProcessor, data: pd.DataFrame):
        st.subheader(processor.get_name())
        st.write(processor.get_description())
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                fill_column = st.selectbox("Выберите столбец для заполнения:", numeric_cols)
            with col2:
                fill_method = st.selectbox("Метод заполнения:", 
                                         ["Среднее значение", "Медиана", "Ноль", "Заданное значение"])
            
            custom_value = None
            if fill_method == "Заданное значение":
                custom_value = st.number_input("Введите значение для заполнения:")
            
            method_map = {
                "Среднее значение": "mean",
                "Медиана": "median", 
                "Ноль": "zero",
                "Заданное значение": "custom"
            }
            
            col1, col2 = st.columns([3, 1])
            with col1:
                missing_count = data[fill_column].isnull().sum()
                st.info(f"Пропусков в столбце '{fill_column}': {missing_count}")
            with col2:
                if st.button("Заполнить пропуски", use_container_width=True):
                    return processor.process(data, column=fill_column, 
                                           method=method_map[fill_method], 
                                           custom_value=custom_value)
        else:
            st.info("В данных нет числовых столбцов для заполнения пропусков")
        
        return data
    
    def _ui_categorical(self, processor: DataProcessor, data: pd.DataFrame):
        st.subheader(processor.get_name())
        st.write(processor.get_description())
        
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            selected_cat_col = st.selectbox("Выберите категориальный столбец:", categorical_cols)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"Уникальных значений в '{selected_cat_col}': {data[selected_cat_col].nunique()}")
                st.write("Примеры значений:", data[selected_cat_col].unique()[:5])
            with col2:
                if st.button("Преобразовать в числовой", use_container_width=True):
                    return processor.process(data, column=selected_cat_col)
        else:
            st.info("В данных нет категориальных столбцов для преобразования")
        
        return data
    
    def _ui_anomaly_detection(self, processor: DataProcessor, data: pd.DataFrame):
        st.subheader(processor.get_name())
        st.write(processor.get_description())
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_num_col = st.selectbox("Выберите числовой столбец для анализа аномалий:", numeric_cols)
            
            z_threshold = st.slider("Порог Z-оценки для обнаружения аномалий:", 
                                   min_value=1.0, max_value=5.0, value=3.0, step=0.1)
            
            # Расчет Z-оценки
            col_data = data[selected_num_col].dropna()
            z_scores = np.abs(stats.zscore(col_data))
            anomalies = col_data[z_scores > z_threshold]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Всего значений", len(col_data))
            with col2:
                st.metric("Найдено аномалий", len(anomalies))
            with col3:
                if len(col_data) > 0:
                    st.metric("Доля аномалий", f"{(len(anomalies) / len(col_data)) * 100:.1f}%")
                else:
                    st.metric("Доля аномалий", "0%")
            
            if len(anomalies) > 0:
                st.write("**Обнаруженные аномалии:**")
                st.dataframe(anomalies, use_container_width=True)
                
                # Визуализация
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.scatter(range(len(col_data)), col_data, c=z_scores, cmap='RdBu_r', alpha=0.6)
                ax.axhline(y=col_data.mean() + z_threshold * col_data.std(), color='r', linestyle='--', label=f'+{z_threshold}σ')
                ax.axhline(y=col_data.mean() - z_threshold * col_data.std(), color='r', linestyle='--', label=f'-{z_threshold}σ')
                ax.set_title(f'Аномалии в столбце {selected_num_col}')
                ax.set_xlabel('Индекс')
                ax.set_ylabel(selected_num_col)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Кнопки для заполнения аномалий
                st.write("**Заполнение аномалий:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Заполнить медианой", use_container_width=True):
                        return processor.process(data, column=selected_num_col, 
                                               threshold=z_threshold, method='median')
                with col2:
                    if st.button("Заполнить средним", use_container_width=True):
                        return processor.process(data, column=selected_num_col,
                                               threshold=z_threshold, method='mean')
                with col3:
                    if st.button("Заполнить модой", use_container_width=True):
                        return processor.process(data, column=selected_num_col,
                                               threshold=z_threshold, method='mode')
            else:
                st.info("Аномалии не обнаружены")
        else:
            st.info("В данных нет числовых столбцов для поиска аномалий")
        
        return data

class MLModelManager:
    def __init__(self):
        self.model_factory = MLModelFactory()
    
    def get_model_ui(self, model_type: str, data: pd.DataFrame):
        model = self.model_factory.create_model(model_type)
        
        if model_type == 'kmeans':
            return self._ui_kmeans(model, data)
        elif model_type == 'dbscan':
            return self._ui_dbscan(model, data)
        elif model_type == 'pca':
            return self._ui_pca(model, data)
        elif model_type == 'tsne':
            return self._ui_tsne(model, data)
        elif model_type in ['random_forest', 'gradient_boosting']:
            return self._ui_classification(model, data)
    
    def _ui_kmeans(self, model: MLModel, data: pd.DataFrame):
        st.write(f"**Настройки {model.get_name()}:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_clusters = st.slider("Количество кластеров", 2, 20, 3, key="kmeans_n_clusters")
            init = st.selectbox("Метод инициализации", ["k-means++", "random"], key="kmeans_init")
        
        with col2:
            max_iter = st.slider("Максимум итераций", 100, 1000, 300, key="kmeans_max_iter")
        
        with col3:
            random_state = st.number_input("Random State", 0, 100, 42, key="kmeans_random_state")
        
        if st.button("Обучить K-Means", type="primary", use_container_width=True):
            try:
                result = model.train(data, n_clusters=n_clusters, init=init, 
                                   max_iter=max_iter, random_state=random_state)
                self._show_clustering_results(data, result, model.get_name())
                st.success(f"K-Means обучен с параметрами: кластеры={n_clusters}")
            except Exception as e:
                st.error(f"Ошибка при обучении K-Means: {e}")
    
    def _ui_dbscan(self, model: MLModel, data: pd.DataFrame):
        st.write(f"**Настройки {model.get_name()}:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            eps = st.slider("EPS (радиус соседства)", 0.1, 2.0, 0.5, 0.1, key="dbscan_eps")
            min_samples = st.slider("Минимальное количество samples", 1, 20, 5, key="dbscan_min_samples")
        
        with col2:
            metric = st.selectbox("Метрика расстояния", 
                                ["euclidean", "manhattan", "cosine"], key="dbscan_metric")
        
        with col3:
            algorithm = st.selectbox("Алгоритм", 
                                   ["auto", "ball_tree", "kd_tree", "brute"], key="dbscan_algorithm")
        
        if st.button("Обучить DBSCAN", type="primary", use_container_width=True):
            try:
                result = model.train(data, eps=eps, min_samples=min_samples, 
                                   metric=metric, algorithm=algorithm)
                self._show_clustering_results(data, result, model.get_name())
                st.success(f"DBSCAN обучен с параметрами: eps={eps}, min_samples={min_samples}")
            except Exception as e:
                st.error(f"Ошибка при обучении DBSCAN: {e}")
    
    def _ui_pca(self, model: MLModel, data: pd.DataFrame):
        st.write(f"**Настройки {model.get_name()}:**")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        max_components = min(10, len(numeric_cols))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_components = st.slider("Количество компонент", 2, max_components, 2, key="pca_n_components")
        
        with col2:
            svd_solver = st.selectbox("SVD решатель", 
                                    ["auto", "full", "arpack", "randomized"], key="pca_svd_solver")
        
        with col3:
            random_state = st.number_input("Random State", 0, 100, 42, key="pca_random_state")
        
        if st.button("Применить PCA", type="primary", use_container_width=True):
            try:
                result = model.train(data, n_components=n_components, 
                                   svd_solver=svd_solver, random_state=random_state)
                self._show_pca_results(result, model.get_name())
                st.success(f"PCA применен с параметрами: компоненты={n_components}")
            except Exception as e:
                st.error(f"Ошибка при применении PCA: {e}")
    
    def _ui_tsne(self, model: MLModel, data: pd.DataFrame):
        st.write(f"**Настройки {model.get_name()}:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_components = st.slider("Количество компонент", 2, 3, 2, key="tsne_n_components")
            perplexity = st.slider("Perplexity", 5, 50, 30, key="tsne_perplexity")
        
        with col2:
            learning_rate = st.slider("Learning Rate", 10, 1000, 200, key="tsne_learning_rate")
            n_iter = st.slider("Количество итераций", 250, 1000, 1000, key="tsne_n_iter")
        
        with col3:
            random_state = st.number_input("Random State", 0, 100, 42, key="tsne_random_state")
            metric = st.selectbox("Метрика расстояния", 
                                ["euclidean", "manhattan", "cosine"], key="tsne_metric")
        
        if st.button("Применить t-SNE", type="primary", use_container_width=True):
            try:
                result = model.train(data, n_components=n_components, perplexity=perplexity,
                                   learning_rate=learning_rate, n_iter=n_iter,
                                   random_state=random_state, metric=metric)
                self._show_tsne_results(result, n_components)
                st.success(f"t-SNE применен с параметрами: компоненты={n_components}")
            except Exception as e:
                st.error(f"Ошибка при применении t-SNE: {e}")
    
    def _ui_classification(self, model: MLModel, data: pd.DataFrame):
        st.write(f"**Настройки {model.get_name()}:**")
        st.info("Реализация классификаторов в разработке")
        
        if st.button(f"Обучить {model.get_name()}", type="primary", use_container_width=True):
            try:
                result = model.train(data)
                st.success(f"{model.get_name()} обучен успешно!")
            except Exception as e:
                st.error(f"Ошибка при обучении {model.get_name()}: {e}")
    
    def _show_clustering_results(self, data: pd.DataFrame, result: Dict, model_name: str):
        st.write(f"**Результаты {model_name}:**")
        
        metrics = result['metrics']
        clusters = result['clusters']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Коэффициент силуэта", f"{metrics['silhouette']:.3f}")
        with col2:
            st.metric("Индекс Calinski-Harabasz", f"{metrics['calinski']:.3f}")
        with col3:
            st.metric("Индекс Davies-Bouldin", f"{metrics['davies']:.3f}")
        with col4:
            st.metric("Количество кластеров", metrics['n_clusters'])
        
        # Визуализация
        X = data[result['features']].fillna(data[result['features']].mean())
        
        if X.shape[1] >= 2:
            from sklearn.decomposition import PCA
            
            if X.shape[1] > 2:
                pca = PCA(n_components=2)
                X_vis = pca.fit_transform(X)
                x_label, y_label = "PC1", "PC2"
            else:
                X_vis = X.values
                x_label, y_label = X.columns[0], X.columns[1]
            
            vis_df = pd.DataFrame({
                x_label: X_vis[:, 0],
                y_label: X_vis[:, 1],
                'Кластер': clusters
            })
            
            fig = px.scatter(
                vis_df, x=x_label, y=y_label, color='Кластер',
                title=f'Визуализация кластеризации {model_name}',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_pca_results(self, result: Dict, model_name: str):
        st.write(f"**Результаты {model_name}:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Объясненная дисперсия", f"{result['total_variance']:.3f}")
        with col2:
            st.metric("Главных компонент", result['transformed_data'].shape[1])
        with col3:
            st.metric("Исходных признаков", len(result['features']))
        
        # Таблица с объясненной дисперсией
        variance_df = pd.DataFrame({
            'Компонента': [f'PC{i+1}' for i in range(len(result['explained_variance']))],
            'Объясненная дисперсия': result['explained_variance'],
            'Накопленная дисперсия': result['cumulative_variance']
        })
        st.write("**Объясненная дисперсия по компонентам:**")
        st.dataframe(variance_df, use_container_width=True)
        
        # Визуализация
        if result['transformed_data'].shape[1] >= 2:
            pca_df = pd.DataFrame({
                'PC1': result['transformed_data'][:, 0],
                'PC2': result['transformed_data'][:, 1]
            })
            
            fig = px.scatter(
                pca_df, x='PC1', y='PC2',
                title='2D проекция данных после PCA'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_tsne_results(self, result: Dict, n_components: int):
        st.write("**Результаты t-SNE:**")
        
        if n_components == 2:
            tsne_df = pd.DataFrame({
                't-SNE 1': result['transformed_data'][:, 0],
                't-SNE 2': result['transformed_data'][:, 1]
            })
            
            fig = px.scatter(
                tsne_df, x='t-SNE 1', y='t-SNE 2',
                title='2D проекция данных после t-SNE'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif n_components == 3:
            tsne_df = pd.DataFrame({
                't-SNE 1': result['transformed_data'][:, 0],
                't-SNE 2': result['transformed_data'][:, 1],
                't-SNE 3': result['transformed_data'][:, 2]
            })
            
            fig = px.scatter_3d(
                tsne_df, x='t-SNE 1', y='t-SNE 2', z='t-SNE 3',
                title='3D проекция данных после t-SNE'
            )
            st.plotly_chart(fig, use_container_width=True)

# Страницы приложения
class BasePage(ABC):
    @abstractmethod
    def render(self):
        pass

class DataUploadPage(BasePage):
    def render(self):
        st.title("Загрузите данные в формате Excel или CSV")
        
        uploaded_file = st.file_uploader(
            "Загрузить данные",
            type=["csv", "xlsx"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                # Здесь должна быть ваша функция load_data из modules.data_loader
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                
                st.session_state["current_data"] = df
                st.session_state["file_name"] = uploaded_file.name
                st.session_state["current_page"] = "analysis"
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Ошибка при загрузке файла: {e}")

class DataAnalysisPage(BasePage):
    def render(self):
        st.title("Анализ данных")
        
        df = st.session_state["current_data"]
        
        # Кнопки анализа (упрощенная версия)
        analysis_options = {
            "Предпросмотр данных": "preview",
            "Описательная статистика": "statistics", 
            "Гистограмма": "histogram",
            "Диаграмма рассеяния": "scatter"
        }
        
        for name, key in analysis_options.items():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.session_state.get(f"show_{key}", False):
                    if st.button(f"Закрыть {name.lower()}", type="secondary", use_container_width=True):
                        st.session_state[f"show_{key}"] = False
                        st.rerun()
                else:
                    if st.button(name, type="primary", use_container_width=True):
                        st.session_state[f"show_{key}"] = True
                        st.rerun()
        
        # Навигация
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Завершить просмотр →", type="primary", use_container_width=True):
                st.session_state["current_page"] = "preprocessing"
                st.rerun()
        
        # Отображение выбранных анализов
        if st.session_state.get("show_preview", False):
            self._show_preview(df)
        if st.session_state.get("show_statistics", False):
            self._show_statistics(df)
    
    def _show_preview(self, df):
        st.subheader("Предпросмотр данных")
        st.dataframe(df.head(10), use_container_width=True)
    
    def _show_statistics(self, df):
        st.subheader("Описательная статистика")
        st.dataframe(df.describe(), use_container_width=True)

class DataPreprocessingPage(BasePage):
    def __init__(self):
        self.data_manager = DataManager()
    
    def render(self):
        st.title("Предобработка данных")
        
        df = st.session_state["current_data"]
        
        # Информация о данных
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Строки", df.shape[0])
        with col2:
            st.metric("Столбцы", df.shape[1])
        with col3:
            st.metric("Пропуски", df.isnull().sum().sum())
        with col4:
            st.metric("Размер", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Кнопки методов предобработк
        
        for id, model, settings in repository_processing.dictionary.items():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.session_state.get(f"show_{key}", False):
                    if st.button(f"Закрыть {name.lower()}", type="secondary", use_container_width=True):
                        st.session_state[f"show_{key}"] = False
                        st.rerun()
                else:
                    if st.button(name, type="primary", use_container_width=True):
                        st.session_state[f"show_{key}"] = True
                        st.rerun()
        
        # Навигация
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Обучение моделей →", type="primary", use_container_width=True):
                st.session_state["current_page"] = "modeling"
                st.rerun()
            
            if st.button("← Назад к анализу", type="primary", use_container_width=True):
                st.session_state["current_page"] = "analysis"
                st.rerun()
        
        # Отображение выбранных методов
        for key in processors.values():
            if st.session_state.get(f"show_{key}", False):
                new_data = self.data_manager.get_processor_ui(key, df)
                if new_data is not None and not new_data.equals(df):
                    st.session_state["current_data"] = new_data
                    st.rerun()

class ModelTrainingPage(BasePage):
    def __init__(self):
        self.model_manager = MLModelManager()
        self.model_factory = MLModelFactory()
    
    def render(self):
        st.title("Обучение моделей")
        
        df = st.session_state["current_data"]
        
        # Информация о данных
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Строки", df.shape[0])
        with col2:
            st.metric("Столбцы", df.shape[1])
        with col3:
            st.metric("Числовые столбцы", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Категориальные столбцы", len(df.select_dtypes(include=['object', 'category']).columns))
        
        # Категории моделей
        categories = {
            "Кластеризация": "clustering",
            "Классификация": "classification", 
            "Понижение размерности": "dimensionality_reduction"
        }
        
        for name, category in categories.items():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.session_state.get(f"show_{category}", False):
                    if st.button(f"Закрыть {name.lower()}", type="secondary", use_container_width=True):
                        st.session_state[f"show_{category}"] = False
                        st.rerun()
                else:
                    if st.button(name, type="primary", use_container_width=True):
                        st.session_state[f"show_{category}"] = True
                        st.rerun()
        
        # Навигация
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("← Назад к предобработке", type="primary", use_container_width=True):
                st.session_state["current_page"] = "preprocessing"
                st.rerun()
        
        # Отображение выбранных категорий и моделей
        for category in categories.values():
            if st.session_state.get(f"show_{category}", False):
                self._show_models_by_category(category, df)
    
    def _show_models_by_category(self, category: str, data: pd.DataFrame):
        st.subheader(f"Модели {category}")
        
        models = self.model_factory.get_models_by_category(category)
        
        for model_type in models:
            model = self.model_factory.create_model(model_type) #
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.session_state.get(f"show_{model_type}", False):
                    if st.button(f"Закрыть {model.get_name()}", type="secondary", use_container_width=True):
                        st.session_state[f"show_{model_type}"] = False
                        st.rerun()
                else:
                    if st.button(model.get_name(), type="primary", use_container_width=True):
                        st.session_state[f"show_{model_type}"] = True
                        st.rerun()
            
            if st.session_state.get(f"show_{model_type}", False):
                self.model_manager.get_model_ui(model_type, data)

# Главное приложение
class DataAnalysisApp:
    def __init__(self):
        self.pages = {
            "upload": DataUploadPage(),
            "analysis": DataAnalysisPage(),
            "preprocessing": DataPreprocessingPage(),
            "modeling": ModelTrainingPage()
        }
        
        # Инициализация состояния сессии
        if "current_data" not in st.session_state:
            st.session_state["current_data"] = None
        if "file_name" not in st.session_state:
            st.session_state["file_name"] = None
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "upload"
    
    def run(self):
        current_page = st.session_state["current_page"]
        
        if current_page in self.pages:
            self.pages[current_page].render()
        else:
            st.error("Страница не найдена")

# Запуск приложения
if __name__ == "__main__":
    app = DataAnalysisApp()
    app.run()