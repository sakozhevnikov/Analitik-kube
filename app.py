import streamlit as st  # Библиотека для веба
import pandas as pd  # Для работы с данными
import numpy as np
from modules.data_loader import load_data # Используем файл из теста


# Настройка страницы
st.set_page_config(
    page_title="Моя ВКР - Аналитическая система", page_icon="🎓", layout="wide"
)


def main():
    st.title("🎓 Моя выпускная квалификационная работа")
    st.markdown("### Аналитическая система для обработки данных")

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

    # Базовая статистика
    st.subheader("📈 Описательная статистика")

    # Показываем только числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe())
    else:
        st.warning("В данных нет числовых колонок для анализа")


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


if __name__ == "__main__":
    main()
