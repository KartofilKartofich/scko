import streamlit as st
import pandas as pd
import numpy as np

# Фиксированные словари данных
DATA_DICTS = st.secrets["all"]

def detect_pos_by_russian(word):
    if not isinstance(word, str) or not word:
        return "NOUN"
    w = word.strip().lower()
    if w.endswith(("ить", "еть", "ать", "ять", "уть", "ють", "ться", "ти", "тись")):
        return "VERB"
    if w.endswith(("ый", "ий", "ой", "ая", "яя", "ое", "ее")):
        return "ADJECTIVE"
    return "NOUN"

def get_filtered_types(pos, translation):
    all_types = list(DATA_DICTS[pos].keys())
    if pos != "NOUN" or not isinstance(translation, str) or not translation:
        return all_types
    
    t = translation.strip().lower()
    if t.endswith(("о", "е")):
        filtered = [x for x in all_types if "NEUT" in x]
    elif t.endswith(("а", "я")):
        filtered = [x for x in all_types if "FEM" in x]
    elif t.endswith("р"):
        filtered = ["NOUN_FEM_3", "NOUN_MASC_1", "NOUN_MASC_2"]
    else:
        filtered = [x for x in all_types if "MASC" in x]

    for i in ["NOUN_IMMUT", "NOUN_NEUT_plur", "NOUN_NEUT_sing"]:
        if i not in filtered: 
            filtered.append(i)

    filtered.append("============")
    for i in all_types:
        if i not in filtered: 
            filtered.append(i)    
        
    return filtered if filtered else all_types

# Функция умного согласования зависимого типа на основе главного типа
def get_agreed_dep_types(dep_pos, main_type):
    all_dep_types = list(DATA_DICTS[dep_pos].keys())
    if not main_type:
        return all_dep_types
    
    # Определяем род из названия основного склонения (MASC, FEM, NEUT)
    gender = "MASC"
    if "FEM" in main_type:
        gender = "FEM"
    elif "NEUT" in main_type:
        gender = "NEUT"
    elif "IMMUT" in main_type:
        gender = "NEUT"
        
    # Оставляем только те типы доп. слова, где совпадает род
    filtered = [x for x in all_dep_types if gender in x]
    return filtered if filtered else all_dep_types

# Получение конкретной формы слова по падежу и числу
def get_nominal_word_form(pos, type_name, stem, number, case):
    if not type_name or type_name not in DATA_DICTS[pos] or not stem:
        return ""
    ending = DATA_DICTS[pos][type_name][number].get(case, "")
    return f"{stem}{ending}"

# Построение единой таблицы фраз (Прилагательное + Существительное)
def build_combined_nominal_df(pos_1, type_1, stem_1, pos_2, type_2, stem_2):
    cases = ["nomn", "gent", "datv", "accs", "ablt", "loct"]
    case_labels = {
        "nomn": "Именительный (nomn)", "gent": "Родительный (gent)", 
        "datv": "Дательный (datv)", "accs": "Винительный (accs)", 
        "ablt": "Творительный (ablt)", "loct": "Предложный (loct)"
    }
    
    data = {"Падеж": [case_labels[c] for c in cases], "Ед.ч. (sing)": [], "Мн.ч. (plur)": []}
    
    for c in cases:
        for num in ["sing", "plur"]:
            # Генерируем формы для обоих слов
            w1 = get_nominal_word_form(pos_1, type_1, stem_1, num, c)
            w2 = get_nominal_word_form(pos_2, type_2, stem_2, num, c)
            
            # Собираем вместе (Прилагательное всегда идет первым, если оно есть)
            if pos_1 == "ADJECTIVE" and pos_2 == "NOUN":
                combined = f"{w1} {w2}".strip()
            elif pos_1 == "NOUN" and pos_2 == "ADJECTIVE":
                combined = f"{w2} {w1}".strip()
            else:
                combined = f"{w1} {w2}".strip() # дефолт, если две одинаковые ЧР
                
            data[f"{'Ед.ч.' if num == 'sing' else 'Мн.ч.'} ({num})"].append(combined)
            
    return pd.DataFrame(data).set_index("Падеж")

# Построение таблицы для глаголов (остается без изменений)
def build_verb_df(type_name, stem):
    if not type_name or type_name not in DATA_DICTS["VERB"]:
        return pd.DataFrame()
    rows = ["1 лицо", "2 лицо", "3 лицо"]
    df_data = {"Форма / Лицо": rows, "Прошедшее (past)": ["", "", ""], "Настоящее (pres)": ["", "", ""], "Будущее (fut)": ["", "", ""]}
    p_sing = DATA_DICTS["VERB"][type_name]["past"].get("sing", "")
    p_plur = DATA_DICTS["VERB"][type_name]["past"].get("plur", "")
    p_sing_form = f"{stem}{p_sing}" if stem else p_sing
    p_plur_form = f"{stem}{p_plur}" if stem else p_plur
    
    for idx, person in enumerate(["1", "2", "3"]):
        df_data["Прошедшее (past)"][idx] = f"Ед: {p_sing_form} | Мн: {p_plur_form}"
        for time_key, col_name in [("pres", "Настоящее (pres)"), ("fut", "Будущее (fut)")]:
            s_ending = DATA_DICTS["VERB"][type_name][time_key]["sing"].get(person, "")
            p_ending = DATA_DICTS["VERB"][type_name][time_key]["plur"].get(person, "")
            s_form = s_ending % stem if "%s" in s_ending else f"{stem}{s_ending}"
            p_form = p_ending % stem if "%s" in p_ending else f"{stem}{p_ending}"
            df_data[col_name][idx] = f"Ед: {s_form} | Мн: {p_form}"
            
    return pd.DataFrame(df_data).set_index("Форма / Лицо")

# Интерфейс Streamlit
st.set_page_config(layout="wide")
st.title("📝 Согласованный редактор склонений слов")

if "df" not in st.session_state:
    st.session_state.df = None
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

uploaded_file = st.file_uploader("Загрузите исходный CSV-файл (разделитель ';')", type=["csv"])

if uploaded_file:
    if st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file, sep=";", keep_default_na=False)
        st.session_state.current_idx = 0
    
    df = st.session_state.df
    
    # Поиск пустых строк
    empty_mask = (df["Русский"] == "") | (df["Словенцко"] == "") | (df["Основная форма"] == "") | (df["Тип"] == "")
    empty_indices = df[empty_mask].index.tolist()
    

    if not empty_indices:
        st.success("🎉 Все строки заполнены! Вы можете скачать итоговый файл внизу страницы.")
        current_row_idx = None
    else:
        if st.session_state.current_idx not in empty_indices:
            st.session_state.current_idx = empty_indices[0]
        current_row_idx = st.session_state.current_idx

    if current_row_idx is not None:
        row = df.loc[current_row_idx]
        st.info(f"Строка {current_row_idx + 1} из {len(df)} (Осталось заполнить: {len(empty_indices)})")
        
        # Первая строка ячеек (1-3 ячейки)
        col1, col2, col3 = st.columns(3)
        with col1:
            val_ru = st.text_input("1. Русский", value=str(row["Русский"]))
        with col2:
            val_sl = st.text_input("2. Словенцко (Перевод)", value=str(row["Словенцко"]))
        with col3:
            val_stem = st.text_input("3. Основная форма (Корень)", value=str(row["Основная форма"]))
            
        detected_pos = detect_pos_by_russian(val_ru)
        available_types = get_filtered_types(detected_pos, val_sl)
        
        # Вторая строка ячеек (4-6 ячейки)
        col4, col5, col6 = st.columns(3)
        with col4:
            current_type = str(row["Тип"])
            default_type_idx = 0
            if current_type in available_types:
                default_type_idx = available_types.index(current_type)
            val_type = st.selectbox("4. Тип (Склонение)", options=available_types, index=default_type_idx)
            
        with col5:
            dep_label = "5. Доп. прилагательное (корень)" if detected_pos == "NOUN" else "5. Доп. существительное (корень)"
            val_dep_stem = st.text_input(dep_label, value="")
            
        # 6-я ячейка — Склонение доп. слова с автоматическим согласованием рода
        with col6:
            dep_pos = "ADJECTIVE" if detected_pos == "NOUN" else "NOUN"
            agreed_types = get_agreed_dep_types(dep_pos, val_type)
            val_dep_type = st.selectbox("6. Тип доп. слова", options=agreed_types, index=0)
            
        # Кнопки управления процессом
        btn_col1, btn_col2, _ = st.columns([1, 1, 4])
        with btn_col1:
            if st.button("💾 Сохранить и далее", type="primary"):
                df.at[current_row_idx, "Русский"] = val_ru
                df.at[current_row_idx, "Словенцко"] = val_sl
                df.at[current_row_idx, "Основная форма"] = val_stem
                df.at[current_row_idx, "Тип"] = val_type
                st.session_state.df = df
                st.rerun()
                
        with btn_col2:
            if st.button("➡️ Пропустить строку"):
                current_pos = empty_indices.index(current_row_idx)
                next_pos = (current_pos + 1) % len(empty_indices)
                st.session_state.current_idx = empty_indices[next_pos]
                st.rerun()
                
        # --- Блок единой интерактивной таблицы фраз ---
        st.markdown("---")
        st.subheader("📊 Единая таблица склонения фразы")
        
        if detected_pos == "VERB":
            st.dataframe(build_verb_df(val_type, val_stem), use_container_width=True)
        else:
            if val_dep_stem.strip():
                # Выводим согласованную таблицу фраз (прилагательное + существительное)
                combined_df = build_combined_nominal_df(
                    detected_pos, val_type, val_stem,
                    dep_pos, val_dep_type, val_dep_stem
                )
                st.dataframe(combined_df, use_container_width=True)
            else:
                # Если доп. слова нет, выводим только форму основного слова
                single_df = build_combined_nominal_df(
                    detected_pos, val_type, val_stem,
                    None, None, ""
                )
                st.dataframe(single_df, use_container_width=True)

    # --- Зона скачивания файла (всегда доступна на уровне загруженного файла) ---
    st.markdown("---")
    st.subheader("📥 Выгрузка результатов")
    csv_buffer = st.session_state.df.to_csv(index=False, sep=";")
    st.download_button(
        label="📥 Скачать отредактированный CSV файл",
        data=csv_buffer,
        file_name="updated_words.csv",
        mime="text/csv"
    )
