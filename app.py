import streamlit as st  # Библиотека для веба
import pandas as pd  # Для работы с данными
import numpy as np
from modules.data_loader import load_data # Используем файл из теста


# Настройка страницы
st.set_page_config(
    page_title="Аналитическая система", layout="wide"
)


def main():
    # Боковая панель с навигацией
    st.sidebar.header("Навигация")
    page = st.sidebar.radio(
        "Выберите раздел:", ["Загрузка данных", "Базовый анализ", "Визуализация"]
    )

    if page == "Загрузка данных":
        show_data_upload()
    elif page == "Базовый анализ":
        show_basic_analysis()
    elif page == "Визуализация":
        show_visualization()


def show_data_upload():
    """Показывает интерфейс загрузки данных."""
    st.header("📁 Загрузка данных")

    uploaded_file = st.file_uploader(
        "Загрузите ваш CSV или Excel файл", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:

            df = load_data(uploaded_file)
            if df is None:
                st.error("❌ Ошибка при загрузке файла")
                return

            # Сохраняем в session_state для использования в других функциях
            st.session_state["current_data"] = df
            st.session_state["file_name"] = uploaded_file.name

            st.success(f"✅ Файл '{uploaded_file.name}' успешно загружен!")

            # Показываем превью данных
            st.subheader("Предпросмотр данных")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Строки", df.shape[0])
            with col2:
                st.metric("Столбцы", df.shape[1])
            with col3:
                st.metric("Размер", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

            st.dataframe(df.head(10))

        except Exception as e:
            st.error(f"❌ Ошибка при загрузке файла: {e}")
    else:
        st.info("👆 Пожалуйста, загрузите CSV или Excel файл")


def show_basic_analysis():
    """Показывает базовый анализ данных."""
    st.header("📊 Базовый анализ данных")

    if "current_data" not in st.session_state:
        st.warning("⚠️ Сначала загрузите данные в разделе 'Загрузка данных'")
        return

    df = st.session_state["current_data"]

    # Кнопка для вызова функции статистики
    if st.button("📊 Показать полную описательную статистику", type="primary"):
        show_descriptive_statistics(df)

def show_visualization():
    """Показывает визуализации данных."""

    if "current_data" not in st.session_state:
        st.warning("⚠️ Сначала загрузите данные в разделе 'Загрузка данных'")
        return

    df = st.session_state["current_data"]

    # Используем переменную df чтобы избежать ошибки
    st.info(
        f"📊 Готово к визуализации! Данные: {df.shape[0]} строк, {df.shape[1]} столбцов" 
    )

    # Мини-превью данных
    with st.expander("🔍 Быстрый просмотр данных"):
        st.dataframe(df.head(3))

    st.write("🚧 Расширенные визуализации будут добавлены в следующем обновлении")

def show_descriptive_statistics(df):
    """
    Показывает расширенную описательную статистику для DataFrame.
    Вызывается по нажатию кнопки.
    
    Args:
        df (pd.DataFrame): DataFrame для анализа
    """
    st.write("### 📊 Полная описательная статистика")
    
    # Для числовых колонок
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.write("#### 🔢 Числовые столбцы")
        numeric_stats = df[numeric_cols].describe()
        
        # Добавляем дополнительные метрики для числовых данных
        additional_numeric_stats = pd.DataFrame({
            'дисперсия': df[numeric_cols].var(),
            'медиана': df[numeric_cols].median(),
            'мода': [df[col].mode().iloc[0] if not df[col].mode().empty else None for col in numeric_cols],
            'количество уникальных': df[numeric_cols].nunique(),
            'количество пропусков': df[numeric_cols].isnull().sum(),
            'скошенность': df[numeric_cols].skew(),
            'эксцесс': df[numeric_cols].kurtosis()
        })
        
        # Объединяем основную и дополнительную статистику
        full_numeric_stats = pd.concat([numeric_stats, additional_numeric_stats], axis=1)
        st.dataframe(full_numeric_stats, use_container_width=True)
    else:
        st.info("ℹ️ В данных нет числовых столбцов")
    
    # Для строковых/категориальных колонок
    string_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(string_cols) > 0:
        st.write("#### 📝 Строковые/категориальные столбцы")
        
        string_stats_data = []
        for col in string_cols:
            value_counts = df[col].value_counts()
            string_stats_data.append({
                'столбец': col,
                'тип данных': str(df[col].dtype),
                'количество уникальных': df[col].nunique(),
                'количество пропусков': df[col].isnull().sum(),
                'самое частые значение': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                'частота самого частого': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'доля самого частого (%)': f"{(value_counts.iloc[0] / len(df) * 100):.1f}%" if len(value_counts) > 0 else "0%",
                'длина макс. значения': df[col].astype(str).str.len().max(),
                'длина мин. значения': df[col].astype(str).str.len().min()
            })
        
        string_stats_df = pd.DataFrame(string_stats_data)
        st.dataframe(string_stats_df, use_container_width=True)
    else:
        st.info("ℹ️ В данных нет строковых/категориальных столбцов")
    
    # Для булевых колонок
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        st.write("#### ⚡ Булевы столбцы")
        
        bool_stats_data = []
        for col in bool_cols:
            value_counts = df[col].value_counts()
            true_count = value_counts.get(True, 0)
            false_count = value_counts.get(False, 0)
            bool_stats_data.append({
                'столбец': col,
                'True значений': true_count,
                'False значений': false_count,
                'количество пропусков': df[col].isnull().sum(),
                'доля True (%)': f"{(true_count / len(df) * 100):.2f}%",
                'доля False (%)': f"{(false_count / len(df) * 100):.2f}%"
            })
        
        bool_stats_df = pd.DataFrame(bool_stats_data)
        st.dataframe(bool_stats_df, use_container_width=True)
    
    # Общая информация о датафрейме
    st.write("### 📋 Общая информация о данных")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Количество строк", len(df))
        st.metric("Количество столбцов", len(df.columns))
    
    with col2:
        st.metric("Общий объем памяти", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        st.metric("Дубликаты", f"{df.duplicated().sum()} строк")
    
    with col3:
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        st.metric("Всего ячеек", total_cells)
        st.metric("Пропущенные ячейки", f"{missing_cells} ({missing_cells/total_cells*100:.1f}%)")
    
    with col4:
        numeric_count = len(numeric_cols)
        string_count = len(string_cols)
        bool_count = len(bool_cols)
        st.metric("Числовые столбцы", numeric_count)
        st.metric("Строковые столбцы", string_count)

if __name__ == "__main__":
    main()
