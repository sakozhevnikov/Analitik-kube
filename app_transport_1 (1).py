import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import os
import zipfile
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Анализатор дорожного трафика",
    page_icon="🚗",
    layout="wide"
)

# Инициализация session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'welcome'
if 'matrices' not in st.session_state:
    st.session_state.matrices = {}
if 'dates' not in st.session_state:
    st.session_state.dates = []
if 'plots_dir' not in st.session_state:
    st.session_state.plots_dir = "traffic_plots"
if not os.path.exists(st.session_state.plots_dir):
    os.makedirs(st.session_state.plots_dir)

def handle_file_upload():
    """Обработка загрузки файла"""
    uploaded_file = st.file_uploader(
        "Загрузите файл с данными",
        type=['csv', 'xlsx'],
        key="file_uploader"
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                        break
                    except:
                        pass
            st.session_state.df = df
            st.session_state.processed = False
            st.session_state.current_page = 'analysis'
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {str(e)}")

def welcome_page():
    """Стартовая страница"""
    st.title("🚗 Анализатор дорожного трафика")
    st.markdown("""---""")
    st.markdown("""
    ### Добро пожаловать!
    Это приложение позволяет анализировать данные о факторах, влияющих на дорожный трафик, и прогнозировать его интенсивность.""")
    st.markdown(""" """)
    st.markdown("""**Поддерживаемые форматы файлов:**""")
    st.markdown(""" - CSV файлы""")
    st.markdown(""" - Excel файлы (.xlsx)""")
    st.markdown(""" """)
    st.markdown("""**Требования к данным:**""")
    st.markdown("""- Данные должны содержать временные метки""")
    st.markdown("""- Должны быть числовые показатели погодных условий""")
    st.markdown("""- Рекомендуется иметь столбцы: время, температура, осадки и так далее.""")
    handle_file_upload()
    st.markdown("""---""")
    with st.expander("📋 Пример структуры данных"):
        st.markdown("""
        | время                | район 1 - район 2 | район 2 - район 3 | интенсивность |
        |----------------------|-------------------|-------------------|---------------|
        | 2024-01-01 08:00:00   | 45                | 552               | низкая         |
        | 2024-01-01 08:15:00   | 78                | 321               | средняя        |
        | 2024-01-01 08:30:00   | 120               | 185               | высокая        |
        """)

def analysis_page():
    """Страница анализа данных"""
    st.title("📊 Анализ данных о трафике")
    st.markdown("""---""")
    if st.session_state.df is None:
        st.warning("Пожалуйста, загрузите данные на стартовой странице")
        if st.button("Вернуться на стартовую страницу"):
            st.session_state.current_page = 'welcome'
            st.rerun()
        return
    df = st.session_state.df
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if st.button("👀 Просмотр данных", use_container_width=True):
            st.session_state.show_data = True
            st.session_state.show_info = False
            st.session_state.show_stats = False
            st.session_state.show_histograms = False
            st.session_state.show_grouped = False
    with col2:
        if st.button("ℹ️ Просмотр информации", use_container_width=True):
            st.session_state.show_data = False
            st.session_state.show_info = True
            st.session_state.show_stats = False
            st.session_state.show_histograms = False
            st.session_state.show_grouped = False
    with col3:
        if st.button("📈 Описательная статистика", use_container_width=True):
            st.session_state.show_data = False
            st.session_state.show_info = False
            st.session_state.show_stats = True
            st.session_state.show_histograms = False
            st.session_state.show_grouped = False
    with col4:
        if st.button("📊 Гистограммы", use_container_width=True):
            st.session_state.show_data = False
            st.session_state.show_info = False
            st.session_state.show_stats = False
            st.session_state.show_histograms = True
            st.session_state.show_grouped = False
    with col5:
        if st.button("⚙️ Подготовка данных", use_container_width=True):
            st.session_state.current_page = 'preprocessing'
            st.rerun()
    with col6:
        if st.button("⏰ Группировка по часу", use_container_width=True):
            st.session_state.show_data = False
            st.session_state.show_info = False
            st.session_state.show_stats = False
            st.session_state.show_histograms = False
            st.session_state.show_grouped = True
    st.markdown("""---""")
    if st.session_state.get('show_data', True):
        show_data_preview(df)
    if st.session_state.get('show_info', False):
        show_data_info(df)
    if st.session_state.get('show_stats', False):
        show_descriptive_stats(df)
    if st.session_state.get('show_histograms', False):
        show_histograms(df)
    if st.session_state.get('show_grouped', False):
        show_grouped_by_hour(df)

def show_data_preview(df):
    """Просмотр данных"""
    st.subheader("Просмотр данных")
    rows_to_show = st.number_input("Количество строк для отображения:", min_value=1, max_value=len(df), value=10)
    st.dataframe(df.head(rows_to_show), use_container_width=True)
    st.info(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")

def show_data_info(df):
    """Информация о данных"""
    st.subheader("Информация о данных")
    st.write("**Типы данных:**")
    dtype_info = pd.DataFrame({
        'Столбец': df.columns,
        'Тип': df.dtypes.values,
        'Ненулевых значений': df.notna().sum().values,
        'Процент заполненности': (df.notna().sum() / len(df) * 100).round(2).values
    })
    st.dataframe(dtype_info, use_container_width=True)
    st.write("**Пропущенные значения:**")
    missing_data = pd.DataFrame({
        'Столбец': df.columns,
        'Пропущенных значений': df.isnull().sum().values,
        'Процент пропусков': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    st.dataframe(missing_data, use_container_width=True)

def show_descriptive_stats(df):
    """Описательная статистика"""
    st.subheader("Описательная статистика")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("В данных нет числовых столбцов для статистического анализа")
        return
    show_all = st.checkbox("Показать статистику по всем числовым столбцам", value=True)
    if show_all:
        stats_df = df[numeric_cols].describe().T
        stats_df['median'] = df[numeric_cols].median()
        stats_df['variance'] = df[numeric_cols].var()
        stats_df = stats_df.round(2)
        stats_df = stats_df.rename(columns={
            'count': 'Количество',
            'mean': 'Среднее',
            'std': 'Станд. отклонение',
            'min': 'Минимум',
            '25%': '25% перцентиль',
            '50%': '50% перцентиль',
            '75%': '75% перцентиль',
            'max': 'Максимум',
            'median': 'Медиана',
            'variance': 'Дисперсия'
        })
        st.dataframe(stats_df, use_container_width=True)
    else:
        selected_cols = st.multiselect(
            "Выберите столбцы для статистики:",
            numeric_cols,
            default=list(numeric_cols[:2]) if len(numeric_cols) >= 2 else list(numeric_cols)
        )
        if selected_cols:
            stats_df = df[selected_cols].describe().T
            stats_df['median'] = df[selected_cols].median()
            stats_df['variance'] = df[selected_cols].var()
            stats_df = stats_df.round(2)
            stats_df = stats_df.rename(columns={
                'count': 'Количество',
                'mean': 'Среднее',
                'std': 'Станд. отклонение',
                'min': 'Минимум',
                '25%': '25% перцентиль',
                '50%': '50% перцентиль',
                '75%': '75% перцентиль',
                'max': 'Максимум',
                'median': 'Медиана',
                'variance': 'Дисперсия'
            })
            st.dataframe(stats_df, use_container_width=True)

def show_histograms(df):
    """Построение гистограмм"""
    st.subheader("Гистограммы")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("Нет числовых столбцов для построения гистограмм")
        return
    selected_columns = st.multiselect(
        "Выберите столбцы для построения гистограмм:",
        numeric_cols,
        default=list(numeric_cols[:2]) if len(numeric_cols) >= 2 else list(numeric_cols)
    )
    if selected_columns:
        bins = st.slider("Количество bins", min_value=5, max_value=100, value=30)
        cols_per_row = 2
        rows = (len(selected_columns) + cols_per_row - 1) // cols_per_row
        for i in range(rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < len(selected_columns):
                    with cols[j]:
                        column = selected_columns[idx]
                        fig, ax = plt.subplots()
                        df[column].hist(bins=bins, ax=ax, alpha=0.7)
                        ax.set_title(f'Гистограмма: {column}')
                        ax.set_xlabel(column)
                        ax.set_ylabel('Частота')
                        st.pyplot(fig)

def show_grouped_by_hour(df):
    """Группировка данных по часу"""
    st.subheader("Группировка данных по часу")
    
    # Находим столбец с временем
    time_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_col = col
            break
    
    if time_col is None:
        # Если не нашли datetime, проверяем строковые столбцы
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Пробуем преобразовать в datetime
                    test_series = pd.to_datetime(df[col], errors='coerce')
                    if not test_series.isna().all():
                        time_col = col
                        df[col] = test_series
                        break
                except:
                    continue
    
    if time_col is None:
        st.error("Не найден столбец с временем. Проверьте данные.")
        return
    
    st.info(f"Найден временной столбец: {time_col}")
    
    # Создаем копию данных для группировки
    df_temp = df.copy()
    
    # Убеждаемся, что временной столбец в правильном формате
    df_temp[time_col] = pd.to_datetime(df_temp[time_col])
    
    # Устанавливаем временной столбец как индекс
    df_temp = df_temp.set_index(time_col)
    
    # Группируем ВЕСЬ dataframe по часу
    try:
        # Для числовых столбцов - среднее значение
        numeric_cols = df_temp.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_numeric = df_temp[numeric_cols].resample('H').mean()
        else:
            df_numeric = pd.DataFrame()
        
        # Для категориальных столбцов - первое значение
        categorical_cols = df_temp.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            df_categorical = df_temp[categorical_cols].resample('H').first()
        else:
            df_categorical = pd.DataFrame()
        
        # Объединяем результаты
        if not df_numeric.empty and not df_categorical.empty:
            df_hourly = pd.concat([df_numeric, df_categorical], axis=1)
        elif not df_numeric.empty:
            df_hourly = df_numeric
        elif not df_categorical.empty:
            df_hourly = df_categorical
        else:
            st.error("Нет данных для группировки")
            return
        
        # Сбрасываем индекс, чтобы вернуть временной столбец обратно в колонки
        df_hourly = df_hourly.reset_index()
        
        # Сохраняем результат
        st.session_state.df = df_hourly
        st.success(f"Данные успешно сгруппированы по часу! Новый размер: {df_hourly.shape[0]} строк, {df_hourly.shape[1]} столбцов")
        
        # Показываем результат
        st.subheader("Результат группировки по часу")
        st.dataframe(df_hourly.head(10), use_container_width=True)
        
        # Не переходим автоматически на подготовку данных
        if st.button("Продолжить анализ сгруппированных данных"):
            st.session_state.show_grouped = False
            st.rerun()
            
    except Exception as e:
        st.error(f"Ошибка при группировке данных: {str(e)}")

def preprocessing_page():
    """Страница подготовки данных"""
    st.title("⚙️ Подготовка данных")
    st.markdown("""---""")
    if st.session_state.df is None:
        st.warning("Нет данных для обработки")
        return
    df = st.session_state.df
    st.subheader("Текущие данные")
    st.dataframe(df.head(), use_container_width=True)
    st.info(f"Текущий размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
    st.markdown("""---""")
    st.subheader("Операции с данными")
    cols_to_drop = st.multiselect(
        "Выберите столбцы для удаления:",
        df.columns,
        key="cols_to_drop"
    )
    if cols_to_drop and st.button("Удалить выбранные столбцы", key="drop_cols"):
        df = df.drop(columns=cols_to_drop)
        st.session_state.df = df
        st.success(f"Удалено {len(cols_to_drop)} столбцов")
        st.rerun()
    col1, col2 = st.columns(2)
    with col1:
        threshold_rows = st.number_input(
            "Процент пропусков для удаления строк (%):",
            min_value=0,
            max_value=100,
            value=30,
            key="threshold_rows"
        )
        if st.button(f"Удалить строки с >{threshold_rows}% пропусков"):
            initial_shape = df.shape
            threshold_value = threshold_rows / 100
            df = df.dropna(thresh=int(df.shape[1] * (1 - threshold_value)))
            st.session_state.df = df
            st.success(f"Удалено {initial_shape[0] - df.shape[0]} строк")
            st.rerun()
    with col2:
        threshold_cols = st.number_input(
            "Процент пропусков для удаления столбцов (%):",
            min_value=0,
            max_value=100,
            value=30,
            key="threshold_cols"
        )
        if st.button(f"Удалить столбцы с >{threshold_cols}% пропусков"):
            initial_shape = df.shape
            threshold_value = threshold_cols / 100
            df = df.dropna(axis=1, thresh=int(df.shape[0] * (1 - threshold_value)))
            st.session_state.df = df
            st.success(f"Удалено {initial_shape[1] - df.shape[1]} столбцов")
            st.rerun()
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Удаление нулевых строк**")
        if st.checkbox("Удалить все нулевые строки", key="drop_all_na_rows"):
            initial_shape = df.shape
            df = df.dropna()
            st.session_state.df = df
            st.success(f"Удалено {initial_shape[0] - df.shape[0]} строк")
            st.rerun()
        else:
            cols_with_missing = [col for col in df.columns if df[col].isnull().sum() > 0]
            if cols_with_missing:
                selected_cols = st.multiselect(
                    "Выберите столбцы для удаления строк с пропусками:",
                    cols_with_missing,
                    key="drop_na_rows_cols"
                )
                if selected_cols and st.button("Удалить строки с пропусками в выбранных столбцах", key="drop_na_rows"):
                    initial_shape = df.shape
                    df = df.dropna(subset=selected_cols)
                    st.session_state.df = df
                    st.success(f"Удалено {initial_shape[0] - df.shape[0]} строк")
                    st.rerun()
    with col2:
        st.write("**Удаление нулевых столбцов**")
        if st.checkbox("Удалить все нулевые столбцы", key="drop_all_na_cols"):
            initial_shape = df.shape
            df = df.dropna(axis=1, how='all')
            st.session_state.df = df
            st.success(f"Удалено {initial_shape[1] - df.shape[1]} столбцов")
            st.rerun()
        else:
            cols_with_all_missing = [col for col in df.columns if df[col].isnull().all()]
            if cols_with_all_missing:
                selected_cols = st.multiselect(
                    "Выберите столбцы для удаления (все значения пропущены):",
                    cols_with_all_missing,
                    key="drop_all_na_cols_select"
                )
                if selected_cols and st.button("Удалить выбранные столбцы", key="drop_all_na_cols_button"):
                    initial_shape = df.shape
                    df = df.drop(columns=selected_cols)
                    st.session_state.df = df
                    st.success(f"Удалено {initial_shape[1] - df.shape[1]} столбцов")
                    st.rerun()
    st.markdown("""---""")
    st.subheader("Заполнение пропусков")
    if st.checkbox("Заполнить пропуски сразу во всех столбцах", key="fill_all"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(numeric_cols) > 0:
            numeric_method = st.radio(
                "Метод для числовых столбцов:",
                ["Среднее", "Медиана", "Константа"],
                key="fill_all_numeric"
            )
            if numeric_method == "Константа":
                const_value = st.number_input("Введите значение для заполнения:", value=0, key="const_all_numeric")
        if len(categorical_cols) > 0:
            cat_method = st.radio(
                "Метод для категориальных столбцов:",
                ["Мода", "Константа"],
                key="fill_all_cat"
            )
            if cat_method == "Константа":
                const_cat_value = st.text_input("Введите значение для заполнения:", value="Unknown", key="const_all_cat")
        if st.button("Заполнить пропуски во всех столбцах", key="fill_all_button"):
            if len(numeric_cols) > 0:
                if numeric_method == "Среднее":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif numeric_method == "Медиана":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif numeric_method == "Константа":
                    df[numeric_cols] = df[numeric_cols].fillna(const_value)
            if len(categorical_cols) > 0:
                if cat_method == "Мода":
                    for col in categorical_cols:
                        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
                elif cat_method == "Константа":
                    for col in categorical_cols:
                        df[col] = df[col].fillna(const_cat_value)
            st.session_state.df = df
            st.success("Пропуски во всех столбцах заполнены")
            st.rerun()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(numeric_cols) > 0:
        st.write("**Заполнение пропусков в числовых столбцах**")
        selected_numeric_cols = st.multiselect(
            "Выберите числовые столбцы для заполнения:",
            numeric_cols,
            default=list(numeric_cols[:2]) if len(numeric_cols) >= 2 else list(numeric_cols),
            key="fill_numeric_cols"
        )
        if selected_numeric_cols:
            fill_method = st.radio(
                "Выберите метод заполнения:",
                ["Среднее", "Медиана", "Константа"],
                key="fill_method_numeric"
            )
            if fill_method == "Константа":
                const_value = st.number_input("Введите значение для заполнения:", value=0, key="const_numeric")
            if st.button("Заполнить пропуски в числовых столбцах", key="fill_numeric"):
                if fill_method == "Среднее":
                    df[selected_numeric_cols] = df[selected_numeric_cols].fillna(df[selected_numeric_cols].mean())
                elif fill_method == "Медиана":
                    df[selected_numeric_cols] = df[selected_numeric_cols].fillna(df[selected_numeric_cols].median())
                elif fill_method == "Константа":
                    df[selected_numeric_cols] = df[selected_numeric_cols].fillna(const_value)
                st.session_state.df = df
                st.success(f"Пропуски в выбранных числовых столбцах заполнены методом: {fill_method}")
                st.rerun()
    if len(categorical_cols) > 0:
        st.write("**Заполнение пропусков в категориальных столбцах**")
        selected_categorical_cols = st.multiselect(
            "Выберите категориальные столбцы для заполнения:",
            categorical_cols,
            default=list(categorical_cols[:1]) if len(categorical_cols) >= 1 else list(categorical_cols),
            key="fill_cat_cols"
        )
        if selected_categorical_cols:
            if st.button("Заполнить пропуски в категориальных столбцах", key="fill_cat"):
                for col in selected_categorical_cols:
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
                st.session_state.df = df
                st.success("Пропуски в категориальных столбцах заполнены модой")
                st.rerun()
    st.markdown("""---""")
    st.subheader("🔍 Поиск аномалий по Z-оценке")
    if st.checkbox("Найти аномалии сразу во всех числовых столбцах", key="find_all_outliers"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            z_score_threshold = st.slider(
                "Порог Z-оценки для выбросов:",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                key="z_threshold_all"
            )
            if st.button("Найти выбросы во всех столбцах", key="find_all_outliers_button"):
                all_outliers = {}
                for col in numeric_cols:
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outlier_indices = np.where(z_scores > z_score_threshold)[0]
                    all_outliers[col] = outlier_indices
                for col, indices in all_outliers.items():
                    if len(indices) > 0:
                        st.write(f"**Выбросы в столбце {col}:** {len(indices)} ({len(indices) / len(df[col]) * 100:.2f}%)")
                        fig, ax = plt.subplots()
                        x = np.arange(len(df[col]))
                        y = df[col].values
                        ax.scatter(x, y, color='blue', label='Основные данные', s=10)
                        ax.scatter(indices, y[indices], color='red', label='Выбросы', s=10)
                        ax.axhline(y=df[col].mean(), color='green', linestyle='--', label='Среднее')
                        ax.axhline(y=df[col].mean() + df[col].std(), color='gray', linestyle=':', label='+1σ')
                        ax.axhline(y=df[col].mean() - df[col].std(), color='gray', linestyle=':', label='-1σ')
                        ax.grid(True, linestyle='--', alpha=0.6)
                        ax.set_title(f'Выбросы в столбце {col}')
                        ax.set_xlabel('Индекс строки')
                        ax.set_ylabel(col)
                        ax.legend()
                        st.pyplot(fig)
                        outliers_df = df.iloc[indices][[col]]
                        st.write("Выбросы:")
                        st.dataframe(outliers_df, use_container_width=True)
                        replace_method = st.radio(
                            f"Заменить выбросы в {col} на:",
                            ["Не заменять", "Среднее", "Медиана", "Мода"],
                            key=f"replace_{col}"
                        )
                        if replace_method != "Не заменять":
                            if replace_method == "Среднее":
                                replace_value = df[col].mean()
                            elif replace_method == "Медиана":
                                replace_value = df[col].median()
                            elif replace_method == "Мода":
                                replace_value = df[col].mode()[0]
                            df.loc[indices, col] = replace_value
                            st.session_state.df = df
                            st.success(f"Выбросы в {col} заменены на {replace_method}")
                            st.rerun()
                    else:
                        st.write(f"**Выбросы в столбце {col}:** не найдены")
        else:
            st.warning("Нет числовых столбцов для поиска аномалий")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        selected_col = st.selectbox(
            "Выберите столбец для поиска выбросов:",
            numeric_cols,
            key="outlier_col"
        )
        z_score_threshold = st.slider(
            "Порог Z-оценки для выбросов:",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            key="z_threshold"
        )
        if st.button("Найти выбросы", key="find_outliers"):
            z_scores = np.abs(stats.zscore(df[selected_col].dropna()))
            outlier_indices = np.where(z_scores > z_score_threshold)[0]
            outlier_count = len(outlier_indices)
            st.write(f"Найдено выбросов: {outlier_count} ({outlier_count / len(z_scores) * 100:.2f}%)")
            fig, ax = plt.subplots()
            x = np.arange(len(df[selected_col]))
            y = df[selected_col].values
            ax.scatter(x, y, color='blue', label='Основные данные', s=10)
            ax.scatter(outlier_indices, y[outlier_indices], color='red', label='Выбросы', s=10)
            ax.axhline(y=df[selected_col].mean(), color='green', linestyle='--', label='Среднее')
            ax.axhline(y=df[selected_col].mean() + df[selected_col].std(), color='gray', linestyle=':', label='+1σ')
            ax.axhline(y=df[selected_col].mean() - df[selected_col].std(), color='gray', linestyle=':', label='-1σ')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_title(f'Выбросы в столбце {selected_col}')
            ax.set_xlabel('Индекс строки')
            ax.set_ylabel(selected_col)
            ax.legend()
            st.pyplot(fig)
            outliers_df = df.iloc[outlier_indices][[selected_col]]
            st.write("Выбросы:")
            st.dataframe(outliers_df, use_container_width=True)
            replace_method = st.radio(
                f"Заменить выбросы в {selected_col} на:",
                ["Не заменять", "Среднее", "Медиана", "Мода"],
                key=f"replace_{selected_col}"
            )
            if replace_method != "Не заменять":
                if replace_method == "Среднее":
                    replace_value = df[selected_col].mean()
                elif replace_method == "Медиана":
                    replace_value = df[selected_col].median()
                elif replace_method == "Мода":
                    replace_value = df[selected_col].mode()[0]
                df.loc[outlier_indices, selected_col] = replace_value
                st.session_state.df = df
                st.success(f"Выбросы в {selected_col} заменены на {replace_method}")
                st.rerun()
    st.markdown("""---""")
    st.subheader("🔄 Перевод категориальных в числовые")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        selected_col = st.selectbox(
            "Выберите категориальный столбец:",
            categorical_cols,
            key="encode_col"
        )
        encode_method = st.radio(
            "Выберите метод кодирования:",
            ["One-Hot Encoding", "Label Encoding"],
            key="encode_method"
        )
        if st.button("Применить кодирование", key="apply_encode"):
            if encode_method == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=[selected_col])
                st.session_state.df = df
                st.success(f"Столбец {selected_col} закодирован методом One-Hot Encoding и удалён из исходных данных")
            elif encode_method == "Label Encoding":
                df[selected_col] = df[selected_col].astype('category').cat.codes
                st.session_state.df = df
                st.success(f"Столбец {selected_col} закодирован методом Label Encoding")
            st.rerun()
    st.markdown("""---""")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Назад к анализу", use_container_width=True):
            st.session_state.current_page = 'analysis'
            st.rerun()
    with col3:
        if st.button("Перейти к результатам →", use_container_width=True):
            st.session_state.processed = True
            st.session_state.current_page = 'results'
            st.rerun()
    st.subheader("Обработанные данные")
    st.dataframe(df.head(), use_container_width=True)
    st.info(f"Текущий размер: {df.shape[0]} строк, {df.shape[1]} столбцов")

def results_page():
    """Страница результатов"""
    st.title("📋 Результаты анализа и прогнозирования")
    st.markdown("""---""")
    
    if st.session_state.df is None:
        st.warning("Нет данных для анализа")
        if st.button("Вернуться к загрузке данных"):
            st.session_state.current_page = 'welcome'
            st.rerun()
        return
    
    # Инициализация данных для демонстрации, если их нет
    if not st.session_state.matrices or not st.session_state.dates:
        st.info("Генерируем демонстрационные данные...")
        generate_sample_data()
    
    if not st.session_state.dates:
        st.warning("Нет доступных данных для отображения")
        return
        
    selected_date = st.selectbox(
        "Выберите дату/время:",
        st.session_state.dates,
        format_func=lambda x: x.strftime("%Y-%m-%d %H:%M") if hasattr(x, 'strftime') else str(x)
    )
    
    if selected_date not in st.session_state.matrices:
        st.error("Выбранная дата не найдена в данных")
        return
        
    matrix = st.session_state.matrices[selected_date]
    
    st.subheader(f"Матрица интенсивности движения на {selected_date.strftime('%Y-%m-%d %H:%M') if hasattr(selected_date, 'strftime') else selected_date}")
    st.write("""
    **Легенда:**
    - Y11: район 1 → район 1
    - Y12: район 1 → район 2
    - Y21: район 2 → район 1
    - и т.д.
    """)
    
    # Создаем DataFrame для отображения матрицы
    matrix_df = pd.DataFrame(matrix, 
                           columns=[f"Район {i+1}" for i in range(matrix.shape[1])], 
                           index=[f"Район {i+1}" for i in range(matrix.shape[0])])
    st.dataframe(matrix_df.style.format("{:.0f}"), use_container_width=True)
    
    # Создаем Excel файл со всеми матрицами
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for date, mat in st.session_state.matrices.items():
            df_mat = pd.DataFrame(mat, 
                                columns=[f"Район {i+1}" for i in range(mat.shape[1])], 
                                index=[f"Район {i+1}" for i in range(mat.shape[0])])
            sheet_name = date.strftime("%Y%m%d_%H%M") if hasattr(date, 'strftime') else str(date)[:31]
            df_mat.to_excel(writer, sheet_name=sheet_name)
    
    output.seek(0)
    
    st.download_button(
        label="📥 Скачать все матрицы (Excel)",
        data=output.getvalue(),
        file_name="traffic_matrices.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    
    st.subheader("Построение графиков динамики")
    col1, col2 = st.columns(2)
    with col1:
        i = st.selectbox("Выберите строку (из района):", range(1, matrix.shape[0] + 1), format_func=lambda x: f"Район {x}")
    with col2:
        j = st.selectbox("Выберите столбец (в район):", range(1, matrix.shape[1] + 1), format_func=lambda x: f"Район {x}")
    
    if st.button("Построить график"):
        values = []
        valid_dates = []
        
        for date in st.session_state.dates:
            if date in st.session_state.matrices:
                mat = st.session_state.matrices[date]
                if i-1 < mat.shape[0] and j-1 < mat.shape[1]:
                    values.append(mat[i-1, j-1])
                    valid_dates.append(date)
        
        if values:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(valid_dates, values, marker='o', linewidth=2, markersize=4, label=f"Район {i} → Район {j}")
            ax.set_title(f"Динамика интенсивности движения: Район {i} → Район {j}", fontsize=14)
            ax.set_xlabel("Время", fontsize=12)
            ax.set_ylabel("Интенсивность", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Сохраняем график
            plot_path = os.path.join(st.session_state.plots_dir, f"traffic_{i}_to_{j}.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            st.success(f"График сохранён: {plot_path}")
        else:
            st.warning("Нет данных для построения графика")
    
    # Создаем ZIP архив с графиками
    if os.path.exists(st.session_state.plots_dir) and os.listdir(st.session_state.plots_dir):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in os.listdir(st.session_state.plots_dir):
                file_path = os.path.join(st.session_state.plots_dir, file)
                if os.path.isfile(file_path):
                    zip_file.write(file_path, file)
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="📥 Скачать все графики (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="traffic_plots.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    st.markdown("""---""")
    if st.button("🔄 Начать новый анализ", use_container_width=True):
        st.session_state.df = None
        st.session_state.processed = False
        st.session_state.current_page = 'welcome'
        st.session_state.matrices = {}
        st.session_state.dates = []
        st.rerun()

def generate_sample_data():
    """Генерация демонстрационных данных для раздела результатов"""
    import datetime
    
    # Создаем демонстрационные даты
    base_date = datetime.datetime.now() - datetime.timedelta(days=7)
    dates = [base_date + datetime.timedelta(hours=i) for i in range(24*7)]
    
    # Создаем демонстрационные матрицы 5x5
    matrices = {}
    for i, date in enumerate(dates):
        # Создаем базовую матрицу с некоторой изменчивостью
        base_matrix = np.array([
            [10, 45, 30, 25, 15],
            [40, 5, 55, 35, 20],
            [25, 50, 8, 40, 30],
            [30, 35, 45, 12, 25],
            [20, 25, 35, 30, 10]
        ])
        
        # Добавляем некоторую изменчивость во времени
        time_factor = 0.5 + 0.5 * np.sin(i / 24 * 2 * np.pi)  # Суточные колебания
        noise = np.random.normal(0, 5, base_matrix.shape)
        
        matrix = base_matrix * time_factor + noise
        matrix = np.maximum(matrix, 0)  # Убеждаемся, что нет отрицательных значений
        matrices[date] = matrix
    
    st.session_state.matrices = matrices
    st.session_state.dates = dates

def main():
    if st.session_state.current_page == 'welcome':
        welcome_page()
    elif st.session_state.current_page == 'analysis':
        analysis_page()
    elif st.session_state.current_page == 'preprocessing':
        preprocessing_page()
    elif st.session_state.current_page == 'results':
        results_page()

if __name__ == "__main__":
    main()