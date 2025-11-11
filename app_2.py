import streamlit as st  # Библиотека для веба
import pandas as pd  # Для работы с данными
import numpy as np
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
    
    # Определяем текущую страницу на основе состояния
    if st.session_state["current_data"] is None:
        show_data_upload()
    else:
        show_data_preprocessing()


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
            
            # Сразу перезагружаем страницу для перехода к предобработке
            st.rerun()

        except Exception as e:
            st.error(f"❌ Ошибка при загрузке файла: {e}")


def show_data_preprocessing():
    """Показывает интерфейс для анализа данных."""
    
    # ЗАГОЛОВОК ПО ЦЕНТРУ (ПЕРЕИМЕНОВАНО)
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


def show_data_preview():
    """Показывает предпросмотр загруженных данных."""
    
    df = st.session_state["current_data"]
    
    st.subheader("Предпросмотр данных")
    
    # Метрики данных
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Строки", df.shape[0])
    with col2:
        st.metric("Столбцы", df.shape[1])
    with col3:
        st.metric("Размер", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Таблица данных
    st.dataframe(df.head(10), use_container_width=True)


def show_descriptive_statistics():
    """Показывает описательную статистику данных с выбором типа анализа."""
    
    df = st.session_state["current_data"]
    
    st.subheader("Описательная статистика")
    
    # ВЫБОР ТИПА СТАТИСТИКИ
    analysis_type = st.radio(
        "Выберите тип анализа:",
        ["По всему датасету", "По отдельному столбцу"],
        horizontal=True
    )
    
    if analysis_type == "По всему датасету":
        show_dataset_statistics(df)
    else:
        show_column_statistics(df)


def show_dataset_statistics(df):
    """Показывает статистику по всему датасету."""
    
    # Основная статистика для числовых колонок
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.write("##### Числовые столбцы")
        numeric_stats = df[numeric_cols].describe()
        st.dataframe(numeric_stats, use_container_width=True)
    else:
        st.info("В данных нет числовых столбцов")
    
    # Общая информация о данных
    st.write("##### Общая информация")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Строки", len(df))
        st.metric("Столбцы", len(df.columns))
    
    with col2:
        st.metric("Числовые столбцы", len(numeric_cols))
        st.metric("Дубликаты", df.duplicated().sum())
    
    with col3:
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        st.metric("Всего ячеек", total_cells)
        st.metric("Пропущенные ячейки", missing_cells)
    
    with col4:
        string_cols = df.select_dtypes(include=['object', 'category']).columns
        st.metric("Строковые столбцы", len(string_cols))
        st.metric("Память", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")


def show_column_statistics(df):
    """Показывает статистику по выбранному столбцу."""
    
    # ВЫБОР СТОЛБЦА
    selected_column = st.selectbox(
        "Выберите столбец для анализа:",
        df.columns
    )
    
    if selected_column:
        col_data = df[selected_column]
        col_type = col_data.dtype
        
        st.write(f"##### Статистика для столбца: '{selected_column}'")
        st.write(f"**Тип данных:** {col_type}")
        
        # Разные типы статистики в зависимости от типа данных
        if np.issubdtype(col_type, np.number):
            show_numeric_column_stats(col_data, selected_column)
        else:
            show_categorical_column_stats(col_data, selected_column)


def show_numeric_column_stats(col_data, col_name):
    """Показывает статистику для числового столбца."""
    
    # Основная статистика
    stats = col_data.describe()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Основная статистика:**")
        stats_data = {
            'Метрика': ['Количество', 'Среднее', 'Стандартное отклонение', 'Минимум', 
                       '25% перцентиль', 'Медиана', '75% перцентиль', 'Максимум'],
            'Значение': [
                stats.get('count', 0),
                f"{stats.get('mean', 0):.2f}",
                f"{stats.get('std', 0):.2f}",
                f"{stats.get('min', 0):.2f}",
                f"{stats.get('25%', 0):.2f}",
                f"{stats.get('50%', 0):.2f}",
                f"{stats.get('75%', 0):.2f}",
                f"{stats.get('max', 0):.2f}"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Дополнительная информация:**")
        
        col1_metrics, col2_metrics = st.columns(2)
        
        with col1_metrics:
            st.metric("Уникальных значений", col_data.nunique())
            st.metric("Пропущенных значений", col_data.isnull().sum())
            st.metric("Дисперсия", f"{col_data.var():.2f}")
        
        with col2_metrics:
            st.metric("Скошенность", f"{col_data.skew():.2f}")
            st.metric("Эксцесс", f"{col_data.kurtosis():.2f}")
            st.metric("Сумма", f"{col_data.sum():.2f}")


def show_categorical_column_stats(col_data, col_name):
    """Показывает статистику для категориального столбца."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Основная информация:**")
        
        info_data = {
            'Метрика': ['Количество записей', 'Уникальных значений', 'Пропущенных значений',
                       'Самое частое значение', 'Частота самого частого'],
            'Значение': [
                len(col_data),
                col_data.nunique(),
                col_data.isnull().sum(),
                col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A',
                col_data.value_counts().iloc[0] if len(col_data.value_counts()) > 0 else 0
            ]
        }
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Распределение значений:**")
        
        value_counts = col_data.value_counts()
        top_10_values = value_counts.head(10)
        
        for value, count in top_10_values.items():
            percentage = (count / len(col_data)) * 100
            st.write(f"- {value}: {count} ({percentage:.1f}%)")


def show_histogram():
    """Показывает гистограмму для выбранного числового столбца."""
    
    df = st.session_state["current_data"]
    
    st.subheader("Гистограмма")
    
    # Выбираем только числовые столбцы
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # ВЫБОР СТОЛБЦА ДЛЯ ГИСТОГРАММЫ -
        selected_column = st.selectbox(
            "Выберите числовой столбец для построения гистограммы:",
            numeric_cols
        )
        
        if selected_column:
            col_data = df[selected_column].dropna()  # Убираем пропуски
            
            # НАСТРОЙКИ ГИСТОГРАММЫ
            bins = st.slider(
                "Количество интервалов:",
                min_value=5,
                max_value=50,
                value=20,
                help="Количество столбцов на гистограмме"
            )
            
            # ПОСТРОЕНИЕ ГИСТОГРАММЫ
            if len(col_data) > 0:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.hist(col_data, bins=bins, color='blue', alpha=0.7, edgecolor='black')
                ax.set_title(f'Гистограмма столбца: {selected_column}')
                ax.set_xlabel(selected_column)
                ax.set_ylabel('Частота')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ
                st.write(f"**Информация о данных:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего значений", len(col_data))
                with col2:
                    st.metric("Минимум", f"{col_data.min():.2f}")
                with col3:
                    st.metric("Максимум", f"{col_data.max():.2f}")
            else:
                st.warning("В выбранном столбце нет данных для построения гистограммы")
    else:
        st.info("В данных нет числовых столбцов для построения гистограммы")


def show_scatter_plot():
    """Показывает диаграмму рассеяния для двух числовых столбцов."""
    
    df = st.session_state["current_data"]
    
    st.subheader("Диаграмма рассеяния")
    
    # Выбираем только числовые столбцы
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) >= 2:
        # ВЫБОР СТОЛБЦОВ ДЛЯ ДИАГРАММЫ РАССЕЯНИЯ
        col1, col2 = st.columns(2)
        
        with col1:
            x_column = st.selectbox(
                "Выберите столбец для оси X:",
                numeric_cols
            )
        
        with col2:
            # Исключаем выбранный столбец для оси X из выбора для оси Y
            y_options = [col for col in numeric_cols if col != x_column]
            y_column = st.selectbox(
                "Выберите столбец для оси Y:",
                y_options
            )
        
        if x_column and y_column:
            # Убираем пропуски из обоих столбцов
            clean_data = df[[x_column, y_column]].dropna()
            x_data = clean_data[x_column]
            y_data = clean_data[y_column]
            
            # ПОСТРОЕНИЕ ДИАГРАММЫ РАССЕЯНИЯ
            if len(x_data) > 0 and len(y_data) > 0:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.scatter(x_data, y_data, color='blue', alpha=0.6, s=50)
                ax.set_title(f'Диаграмма рассеяния: {x_column} vs {y_column}')
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ
                st.write("**Информация о данных:**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Всего точек", len(x_data))
                with col2:
                    st.metric("Корреляция", f"{x_data.corr(y_data):.3f}")
                with col3:
                    st.metric(f"Среднее {x_column}", f"{x_data.mean():.2f}")
                with col4:
                    st.metric(f"Среднее {y_column}", f"{y_data.mean():.2f}")
                
                # Информация о выбросах
                st.write("**Описательная статистика:**")
                stats_df = pd.DataFrame({
                    x_column: x_data.describe(),
                    y_column: y_data.describe()
                })
                st.dataframe(stats_df, use_container_width=True)
                
            else:
                st.warning("Недостаточно данных для построения диаграммы рассеяния")
    else:
        st.info("Для построения диаграммы рассеяния нужно как минимум 2 числовых столбца")


if __name__ == "__main__":
    main()