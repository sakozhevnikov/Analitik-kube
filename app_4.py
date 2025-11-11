import streamlit as st  # Библиотека для веба
import pandas as pd  # Для работы с данными
import numpy as np
from scipy import stats  # Для z-оценки
from modules.data_loader import load_data # Используем файл из теста
import matplotlib.pyplot as plt  # Добавляем импорт для графиков


# Настройка страницы
st.set_page_config(
    page_title="Аналитическая система", layout="wide"
)


def main():
    # Инициализация состояния сессии
    if "current_data" not in st.session_state:
        st.session_state["current_data"] = None
    if "file_name" not in st.session_state:
        st.session_state["file_name"] = None
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "upload"  # upload, analysis, preprocessing, modeling
    
    # Определяем текущую страницу на основе состояния
    if st.session_state["current_page"] == "upload":
        show_data_upload()
    elif st.session_state["current_page"] == "analysis":
        show_data_analysis()
    elif st.session_state["current_page"] == "preprocessing":
        show_data_preprocessing()
    else:  # modeling
        show_model_training()


def show_data_upload():
    """Минималистичный интерфейс загрузки данных."""
    
    # ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННОЙ
    uploaded_file = None
    
    # Создаем 3 колонки с соотношением ширины 1:2:1
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Работаем только со средней колонкой (col2)
    with col2:
        # Выводим заголовок первого уровня
        st.title("Загрузите данные в формате Excel или CSV")
        
        # Создаем виджет для загрузки файлов
        uploaded_file = st.file_uploader(
            "Загрузить данные",
            type=["csv", "xlsx"],
            label_visibility="collapsed"
        )
    
    # ОБРАБОТКА ЗАГРУЖЕННОГО ФАЙЛА
    if uploaded_file is not None and st.session_state["current_data"] is None:
        try:
            df = load_data(uploaded_file)
            
            if df is None:
                st.error("❌ Ошибка при загрузке файла")
                return

            # СОХРАНЕНИЕ ДАННЫХ В СЕССИИ
            st.session_state["current_data"] = df
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["current_page"] = "analysis"
            
            # Сразу перезагружаем страницу для перехода к анализу
            st.rerun()

        except Exception as e:
            st.error(f"❌ Ошибка при загрузке файла: {e}")


def show_data_analysis():
    """Показывает интерфейс для анализа данных."""
    
    # ЗАГОЛОВОК ПО ЦЕНТРУ
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.title("Анализ данных")
    
    # КНОПКА ПРЕДПРОСМОТРА ДАННЫХ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("show_preview", False):
            if st.button("Закрыть предпросмотр", type="secondary", use_container_width=True):
                st.session_state["show_preview"] = False
                st.rerun()
        else:
            if st.button("Предпросмотр данных", type="primary", use_container_width=True):
                st.session_state["show_preview"] = True
                st.rerun()
    
    # КНОПКА ОПИСАТЕЛЬНОЙ СТАТИСТИКИ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("show_statistics", False):
            if st.button("Закрыть статистику", type="secondary", use_container_width=True):
                st.session_state["show_statistics"] = False
                st.rerun()
        else:
            if st.button("Описательная статистика", type="primary", use_container_width=True):
                st.session_state["show_statistics"] = True
                st.rerun()
    
    # КНОПКА ГИСТОГРАММЫ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("show_histogram", False):
            if st.button("Закрыть гистограмму", type="secondary", use_container_width=True):
                st.session_state["show_histogram"] = False
                st.rerun()
        else:
            if st.button("Гистограмма", type="primary", use_container_width=True):
                st.session_state["show_histogram"] = True
                st.rerun()
    
    # КНОПКА ДИАГРАММЫ РАССЕЯНИЯ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("show_scatter", False):
            if st.button("Закрыть диаграмму рассеяния", type="secondary", use_container_width=True):
                st.session_state["show_scatter"] = False
                st.rerun()
        else:
            if st.button("Диаграмма рассеяния", type="primary", use_container_width=True):
                st.session_state["show_scatter"] = True
                st.rerun()
    
    # КНОПКА ЗАВЕРШИТЬ ПРОСМОТР И ПЕРЕЙТИ К ПРЕДОБРАБОТКЕ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Завершить просмотр →", type="primary", use_container_width=True):
            st.session_state["current_page"] = "preprocessing"
            st.rerun()
    
    # Показываем предпросмотр если включен
    if st.session_state.get("show_preview", False):
        show_data_preview()
    
    # Показываем статистику если включена
    if st.session_state.get("show_statistics", False):
        show_descriptive_statistics()
    
    # Показываем гистограмму если включена
    if st.session_state.get("show_histogram", False):
        show_histogram()
    
    # Показываем диаграмму рассеяния если включена
    if st.session_state.get("show_scatter", False):
        show_scatter_plot()


def show_data_preprocessing():
    """Страница для предобработки данных с сворачиваемыми разделами."""
    
    # ЗАГОЛОВОК ПО ЦЕНТРУ
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.title("Предобработка данных")
    
    df = st.session_state["current_data"]
    
    # Информация о текущих данных
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Строки", df.shape[0])
    with col2:
        st.metric("Столбцы", df.shape[1])
    with col3:
        st.metric("Пропуски", df.isnull().sum().sum())
    with col4:
        st.metric("Размер", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # КНОПКА УДАЛЕНИЯ НУЛЕВЫХ СТРОК
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("show_remove_nulls", False):
            if st.button("Закрыть удаление нулевых строк", type="secondary", use_container_width=True):
                st.session_state["show_remove_nulls"] = False
                st.rerun()
        else:
            if st.button("Удалить нулевые строки", type="primary", use_container_width=True):
                st.session_state["show_remove_nulls"] = True
                st.rerun()
    
    # КНОПКА ЗАПОЛНЕНИЯ ПРОПУСКОВ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("show_fill_nulls", False):
            if st.button("Закрыть заполнение пропусков", type="secondary", use_container_width=True):
                st.session_state["show_fill_nulls"] = False
                st.rerun()
        else:
            if st.button("Заполнить пропуски", type="primary", use_container_width=True):
                st.session_state["show_fill_nulls"] = True
                st.rerun()
    
    # КНОПКА ПРЕОБРАЗОВАНИЯ КАТЕГОРИАЛЬНЫХ ДАННЫХ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("show_categorical", False):
            if st.button("Закрыть преобразование категорий", type="secondary", use_container_width=True):
                st.session_state["show_categorical"] = False
                st.rerun()
        else:
            if st.button("Преобразовать категории", type="primary", use_container_width=True):
                st.session_state["show_categorical"] = True
                st.rerun()
    
    # КНОПКА ПОИСКА АНОМАЛИЙ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("show_anomalies", False):
            if st.button("Закрыть поиск аномалий", type="secondary", use_container_width=True):
                st.session_state["show_anomalies"] = False
                st.rerun()
        else:
            if st.button("Поиск аномалий", type="primary", use_container_width=True):
                st.session_state["show_anomalies"] = True
                st.rerun()
    
    # КНОПКА ПЕРЕХОДА К ОБУЧЕНИЮ МОДЕЛЕЙ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Обучение моделей →", type="primary", use_container_width=True):
            st.session_state["current_page"] = "modeling"
            st.rerun()
    
    # КНОПКИ НАВИГАЦИИ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Назад к анализу", type="primary", use_container_width=True):
            st.session_state["current_page"] = "analysis"
            st.rerun()
    
    # Показываем соответствующие разделы если они включены
    if st.session_state.get("show_remove_nulls", False):
        show_remove_null_rows()
    
    if st.session_state.get("show_fill_nulls", False):
        show_fill_nulls()
    
    if st.session_state.get("show_categorical", False):
        show_categorical_conversion()
    
    if st.session_state.get("show_anomalies", False):
        show_anomaly_detection()


def show_model_training():
    """Страница для обучения моделей машинного обучения."""
    
    # ЗАГОЛОВОК ПО ЦЕНТРУ
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.title("Обучение моделей")
    
    df = st.session_state["current_data"]
    
    # Информация о текущих данных
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Строки", df.shape[0])
    with col2:
        st.metric("Столбцы", df.shape[1])
    with col3:
        st.metric("Числовые столбцы", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Категориальные столбцы", len(df.select_dtypes(include=['object', 'category']).columns))
    
    # ОСНОВНЫЕ КАТЕГОРИИ МОДЕЛЕЙ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("show_clustering", False):
            if st.button("Закрыть кластеризацию", type="secondary", use_container_width=True):
                st.session_state["show_clustering"] = False
                st.rerun()
        else:
            if st.button("Кластеризация", type="primary", use_container_width=True):
                st.session_state["show_clustering"] = True
                st.rerun()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("show_classification", False):
            if st.button("Закрыть классификацию", type="secondary", use_container_width=True):
                st.session_state["show_classification"] = False
                st.rerun()
        else:
            if st.button("Классификация", type="primary", use_container_width=True):
                st.session_state["show_classification"] = True
                st.rerun()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("show_dimensionality", False):
            if st.button("Закрыть понижение размерности", type="secondary", use_container_width=True):
                st.session_state["show_dimensionality"] = False
                st.rerun()
        else:
            if st.button("Понижение размерности", type="primary", use_container_width=True):
                st.session_state["show_dimensionality"] = True
                st.rerun()
    
    # КНОПКИ НАВИГАЦИИ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Назад к предобработке", type="primary", use_container_width=True):
            st.session_state["current_page"] = "preprocessing"
            st.rerun()
    
    # Показываем соответствующие разделы если они включены
    if st.session_state.get("show_clustering", False):
        show_clustering_models()
    
    if st.session_state.get("show_classification", False):
        show_classification_models()
    
    if st.session_state.get("show_dimensionality", False):
        show_dimensionality_models()


def show_clustering_models():
    """Показывает модели кластеризации."""
    
    st.subheader("Модели кластеризации")
    
    # Кнопки выбора моделей кластеризации
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.get("show_kmeans", False):
            if st.button("Закрыть K-Means", type="secondary", use_container_width=True):
                st.session_state["show_kmeans"] = False
                st.rerun()
        else:
            if st.button("K-Means", type="primary", use_container_width=True):
                st.session_state["show_kmeans"] = True
                st.rerun()
    
    with col2:
        if st.session_state.get("show_dbscan", False):
            if st.button("Закрыть DBSCAN", type="secondary", use_container_width=True):
                st.session_state["show_dbscan"] = False
                st.rerun()
        else:
            if st.button("DBSCAN", type="primary", use_container_width=True):
                st.session_state["show_dbscan"] = True
                st.rerun()
    
    with col3:
        if st.session_state.get("show_hdbscan", False):
            if st.button("Закрыть HDBSCAN", type="secondary", use_container_width=True):
                st.session_state["show_hdbscan"] = False
                st.rerun()
        else:
            if st.button("HDBSCAN", type="primary", use_container_width=True):
                st.session_state["show_hdbscan"] = True
                st.rerun()
    
    # Показываем настройки моделей если они выбраны
    if st.session_state.get("show_kmeans", False):
        show_kmeans_settings()
    
    if st.session_state.get("show_dbscan", False):
        show_dbscan_settings()
    
    if st.session_state.get("show_hdbscan", False):
        show_hdbscan_settings()


def show_classification_models():
    """Показывает модели классификации."""
    
    st.subheader("Модели классификации")
    
    # Кнопки выбора моделей классификации
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.get("show_random_forest", False):
            if st.button("Закрыть Случайный лес", type="secondary", use_container_width=True):
                st.session_state["show_random_forest"] = False
                st.rerun()
        else:
            if st.button("Случайный лес", type="primary", use_container_width=True):
                st.session_state["show_random_forest"] = True
                st.rerun()
    
    with col2:
        if st.session_state.get("show_gradient_boosting", False):
            if st.button("Закрыть Градиентный бустинг", type="secondary", use_container_width=True):
                st.session_state["show_gradient_boosting"] = False
                st.rerun()
        else:
            if st.button("Градиентный бустинг", type="primary", use_container_width=True):
                st.session_state["show_gradient_boosting"] = True
                st.rerun()
    
    # Показываем настройки моделей если они выбраны
    if st.session_state.get("show_random_forest", False):
        show_random_forest_settings()
    
    if st.session_state.get("show_gradient_boosting", False):
        show_gradient_boosting_settings()


def show_dimensionality_models():
    """Показывает модели понижения размерности."""
    
    st.subheader("Модели понижения размерности")
    
    # Кнопки выбора моделей понижения размерности
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.get("show_pca", False):
            if st.button("Закрыть PCA", type="secondary", use_container_width=True):
                st.session_state["show_pca"] = False
                st.rerun()
        else:
            if st.button("PCA", type="primary", use_container_width=True):
                st.session_state["show_pca"] = True
                st.rerun()
    
    with col2:
        if st.session_state.get("show_tsne", False):
            if st.button("Закрыть t-SNE", type="secondary", use_container_width=True):
                st.session_state["show_tsne"] = False
                st.rerun()
        else:
            if st.button("t-SNE", type="primary", use_container_width=True):
                st.session_state["show_tsne"] = True
                st.rerun()
    
    # Показываем настройки моделей если они выбраны
    if st.session_state.get("show_pca", False):
        show_pca_settings()
    
    if st.session_state.get("show_tsne", False):
        show_tsne_settings()


def show_kmeans_settings():
    """Настройки гиперпараметров для K-Means."""
    
    st.write("**Настройки K-Means:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_clusters = st.slider("Количество кластеров", 2, 20, 3, key="kmeans_n_clusters")
        init = st.selectbox("Метод инициализации", ["k-means++", "random"], key="kmeans_init")
    
    with col2:
        max_iter = st.slider("Максимум итераций", 100, 1000, 300, key="kmeans_max_iter")
        tol = st.number_input("Допуск сходимости", 1e-5, 1e-1, 1e-4, format="%f", key="kmeans_tol")
    
    with col3:
        algorithm = st.selectbox("Алгоритм", ["auto", "full", "elkan"], key="kmeans_algorithm")
        random_state = st.number_input("Random State", 0, 100, 42, key="kmeans_random_state")
    
    if st.button("Обучить K-Means", type="primary", use_container_width=True):
        st.success(f"K-Means обучен с параметрами: кластеры={n_clusters}, итерации={max_iter}")
        # Здесь будет код для обучения модели


def show_dbscan_settings():
    """Настройки гиперпараметров для DBSCAN."""
    
    st.write("**Настройки DBSCAN:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        eps = st.slider("EPS (радиус соседства)", 0.1, 2.0, 0.5, 0.1, key="dbscan_eps")
        min_samples = st.slider("Минимальное количество samples", 1, 20, 5, key="dbscan_min_samples")
    
    with col2:
        metric = st.selectbox("Метрика расстояния", 
                            ["euclidean", "manhattan", "cosine"], key="dbscan_metric")
        algorithm = st.selectbox("Алгоритм", 
                               ["auto", "ball_tree", "kd_tree", "brute"], key="dbscan_algorithm")
    
    with col3:
        leaf_size = st.slider("Leaf Size", 10, 50, 30, key="dbscan_leaf_size")
    
    if st.button("Обучить DBSCAN", type="primary", use_container_width=True):
        st.success(f"DBSCAN обучен с параметрами: eps={eps}, min_samples={min_samples}")
        # Здесь будет код для обучения модели


def show_hdbscan_settings():
    """Настройки гиперпараметров для HDBSCAN."""
    
    st.write("**Настройки HDBSCAN:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_cluster_size = st.slider("Минимальный размер кластера", 2, 50, 5, key="hdbscan_min_cluster_size")
        min_samples = st.slider("Минимальное количество samples", 1, 20, 5, key="hdbscan_min_samples")
    
    with col2:
        cluster_selection_epsilon = st.slider("Cluster Selection Epsilon", 0.0, 1.0, 0.0, 0.1, 
                                            key="hdbscan_epsilon")
        alpha = st.slider("Alpha", 0.1, 2.0, 1.0, 0.1, key="hdbscan_alpha")
    
    with col3:
        metric = st.selectbox("Метрика расстояния", 
                            ["euclidean", "manhattan", "cosine"], key="hdbscan_metric")
        cluster_selection_method = st.selectbox("Метод выбора кластера", 
                                              ["eom", "leaf"], key="hdbscan_method")
    
    if st.button("Обучить HDBSCAN", type="primary", use_container_width=True):
        st.success(f"HDBSCAN обучен с параметрами: min_cluster_size={min_cluster_size}")
        # Здесь будет код для обучения модели


def show_random_forest_settings():
    """Настройки гиперпараметров для Random Forest."""
    
    st.write("**Настройки Random Forest:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_estimators = st.slider("Количество деревьев", 10, 500, 100, key="rf_n_estimators")
        max_depth = st.slider("Максимальная глубина", 1, 50, 10, key="rf_max_depth")
    
    with col2:
        min_samples_split = st.slider("Минимальное samples для разделения", 2, 20, 2, key="rf_min_samples_split")
        min_samples_leaf = st.slider("Минимальное samples в листе", 1, 10, 1, key="rf_min_samples_leaf")
    
    with col3:
        max_features = st.selectbox("Максимальное количество features", 
                                  ["auto", "sqrt", "log2"], key="rf_max_features")
        random_state = st.number_input("Random State", 0, 100, 42, key="rf_random_state")
    
    if st.button("Обучить Random Forest", type="primary", use_container_width=True):
        st.success(f"Random Forest обучен с параметрами: деревья={n_estimators}, глубина={max_depth}")
        # Здесь будет код для обучения модели


def show_gradient_boosting_settings():
    """Настройки гиперпараметров для Gradient Boosting."""
    
    st.write("**Настройки Gradient Boosting:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_estimators = st.slider("Количество estimators", 10, 500, 100, key="gb_n_estimators")
        learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01, key="gb_learning_rate")
    
    with col2:
        max_depth = st.slider("Максимальная глубина", 1, 10, 3, key="gb_max_depth")
        min_samples_split = st.slider("Минимальное samples для разделения", 2, 20, 2, key="gb_min_samples_split")
    
    with col3:
        subsample = st.slider("Subsample", 0.1, 1.0, 1.0, 0.1, key="gb_subsample")
        random_state = st.number_input("Random State", 0, 100, 42, key="gb_random_state")
    
    if st.button("Обучить Gradient Boosting", type="primary", use_container_width=True):
        st.success(f"Gradient Boosting обучен с параметрами: estimators={n_estimators}, learning_rate={learning_rate}")
        # Здесь будет код для обучения модели


def show_pca_settings():
    """Настройки гиперпараметров для PCA."""
    
    st.write("**Настройки PCA:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_components = st.slider("Количество компонент", 2, 20, 2, key="pca_n_components")
    
    with col2:
        svd_solver = st.selectbox("SVD решатель", 
                                ["auto", "full", "arpack", "randomized"], key="pca_svd_solver")
    
    with col3:
        random_state = st.number_input("Random State", 0, 100, 42, key="pca_random_state")
    
    if st.button("Применить PCA", type="primary", use_container_width=True):
        st.success(f"PCA применен с параметрами: компоненты={n_components}")
        # Здесь будет код для применения PCA


def show_tsne_settings():
    """Настройки гиперпараметров для t-SNE."""
    
    st.write("**Настройки t-SNE:**")
    
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
        st.success(f"t-SNE применен с параметрами: компоненты={n_components}, perplexity={perplexity}")
        # Здесь будет код для применения t-SNE


# ОСТАВШИЕСЯ ФУНКЦИИ ПРЕДОБРАБОТКИ И АНАЛИЗА (без изменений)
def show_remove_null_rows():
    """Показывает интерфейс для удаления нулевых строк."""
    
    st.subheader("Удаление нулевых строк")
    
    df = st.session_state["current_data"]
    
    st.write("Удалить строки, содержащие пропущенные значения")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Текущее количество строк с пропусками: {df.isnull().any(axis=1).sum()}")
        st.info(f"Всего строк в данных: {df.shape[0]}")
    with col2:
        if st.button("Удалить нулевые строки", use_container_width=True):
            initial_rows = df.shape[0]
            df_cleaned = df.dropna()
            st.session_state["current_data"] = df_cleaned
            st.success(f"Удалено {initial_rows - df_cleaned.shape[0]} строк с пропусками")
            st.rerun()


def show_fill_nulls():
    """Показывает интерфейс для заполнения пропусков."""
    
    st.subheader("Заполнение пропусков")
    
    df = st.session_state["current_data"]
    st.write("Заполнить пропущенные значения в числовых столбцах")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            fill_column = st.selectbox("Выберите столбец для заполнения:", numeric_cols)
        with col2:
            fill_method = st.selectbox("Метод заполнения:", 
                                     ["Среднее значение", "Медиана", "Ноль", "Заданное значение"])
        
        if fill_method == "Заданное значение":
            custom_value = st.number_input("Введите значение для заполнения:")
        else:
            custom_value = None
        
        col1, col2 = st.columns([3, 1])
        with col1:
            missing_count = df[fill_column].isnull().sum()
            st.info(f"Пропусков в столбце '{fill_column}': {missing_count}")
            if not df[fill_column].isnull().all():
                st.info(f"Текущие статистики: Среднее = {df[fill_column].mean():.2f}, Медиана = {df[fill_column].median():.2f}")
        with col2:
            if st.button("Заполнить пропуски", use_container_width=True):
                if fill_method == "Среднее значение":
                    fill_value = df[fill_column].mean()
                elif fill_method == "Медиана":
                    fill_value = df[fill_column].median()
                elif fill_method == "Ноль":
                    fill_value = 0
                else:
                    fill_value = custom_value
                
                df_filled = df.copy()
                df_filled[fill_column] = df_filled[fill_column].fillna(fill_value)
                st.session_state["current_data"] = df_filled
                st.success(f"Пропуски в столбце '{fill_column}' заполнены значением: {fill_value:.2f}")
                st.rerun()
    else:
        st.info("В данных нет числовых столбцов для заполнения пропусков")


def show_categorical_conversion():
    """Показывает интерфейс для преобразования категориальных данных."""
    
    st.subheader("Преобразование категориальных данных")
    
    df = st.session_state["current_data"]
    st.write("Преобразовать категориальный столбец в числовой с помощью one-hot encoding")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        selected_cat_col = st.selectbox("Выберите категориальный столбец:", categorical_cols)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Уникальных значений в '{selected_cat_col}': {df[selected_cat_col].nunique()}")
            st.write("Примеры значений:", df[selected_cat_col].unique()[:5])
            
            # Показываем распределение значений
            value_counts = df[selected_cat_col].value_counts()
            st.write("**Распределение значений:**")
            for value, count in value_counts.head(5).items():
                percentage = (count / len(df)) * 100
                st.write(f"- {value}: {count} ({percentage:.1f}%)")
                
        with col2:
            if st.button("Преобразовать в числовой", use_container_width=True):
                # One-hot encoding
                dummies = pd.get_dummies(df[selected_cat_col], prefix=selected_cat_col)
                df_encoded = pd.concat([df, dummies], axis=1)
                # Удаляем исходный столбец
                df_encoded = df_encoded.drop(columns=[selected_cat_col])
                st.session_state["current_data"] = df_encoded
                st.success(f"Столбец '{selected_cat_col}' преобразован в {dummies.shape[1]} числовых столбцов")
                st.rerun()
    else:
        st.info("В данных нет категориальных столбцов для преобразования")


def show_anomaly_detection():
    """Показывает интерфейс для поиска аномалий с помощью Z-оценки."""
    
    st.subheader("Поиск аномалий (Z-оценка)")
    
    df = st.session_state["current_data"]
    st.write("Обнаружение выбросов с использованием Z-оценки")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        selected_num_col = st.selectbox("Выберите числовой столбец для анализа аномалий:", numeric_cols)
        
        z_threshold = st.slider("Порог Z-оценки для обнаружения аномалий:", 
                               min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        
        # Расчет Z-оценки
        col_data = df[selected_num_col].dropna()
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
            
            # КНОПКИ ДЛЯ ЗАПОЛНЕНИЯ АНОМАЛИЙ
            st.write("**Заполнение аномалий:**")
            st.write("Заменить обнаруженные аномалии выбранным значением")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Заполнить медианой", use_container_width=True):
                    fill_anomalies_with_method(selected_num_col, anomalies, "median")
                    
            with col2:
                if st.button("Заполнить средним", use_container_width=True):
                    fill_anomalies_with_method(selected_num_col, anomalies, "mean")
                    
            with col3:
                if st.button("Заполнить модой", use_container_width=True):
                    fill_anomalies_with_method(selected_num_col, anomalies, "mode")
            
            # Дополнительная информация об аномалиях
            st.write("**Статистики аномалий:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Мин. аномалия", f"{anomalies.min():.2f}")
            with col2:
                st.metric("Макс. аномалия", f"{anomalies.max():.2f}")
            with col3:
                st.metric("Среднее аномалий", f"{anomalies.mean():.2f}")
            with col4:
                st.metric("Медиана аномалий", f"{anomalies.median():.2f}")
                
        else:
            st.info("Аномалии не обнаружены")
    else:
        st.info("В данных нет числовых столбцов для поиска аномалий")


def fill_anomalies_with_method(column_name, anomalies, method):
    """Заполняет аномалии выбранным методом."""
    
    df = st.session_state["current_data"]
    
    # Получаем индексы аномалий
    col_data = df[column_name].dropna()
    z_scores = np.abs(stats.zscore(col_data))
    z_threshold = st.session_state.get("z_threshold", 3.0)
    anomaly_indices = col_data[z_scores > z_threshold].index
    
    # Вычисляем значение для заполнения
    if method == "median":
        fill_value = df[column_name].median()
        method_name = "медианой"
    elif method == "mean":
        fill_value = df[column_name].mean()
        method_name = "средним"
    elif method == "mode":
        fill_value = df[column_name].mode()[0] if not df[column_name].mode().empty else df[column_name].median()
        method_name = "модой"
    else:
        fill_value = df[column_name].median()
        method_name = "медианой"
    
    # Заполняем аномалии
    df_filled = df.copy()
    df_filled.loc[anomaly_indices, column_name] = fill_value
    st.session_state["current_data"] = df_filled
    
    st.success(f"Аномалии в столбце '{column_name}' заполнены {method_name} ({fill_value:.2f})")
    st.rerun()


# ОСТАВШИЕСЯ ФУНКЦИИ БЕЗ ИЗМЕНЕНИЙ (show_data_preview, show_descriptive_statistics, и т.д.)
# ... (все остальные функции остаются без изменений)

if __name__ == "__main__":
    main()