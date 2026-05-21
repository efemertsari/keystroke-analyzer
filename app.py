import streamlit as st
import pandas as pd
import numpy as np
import time
from pynput import keyboard
import threading
import altair as alt

# --- MODELLER ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # Neural Network Kütüphanesi
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Biometric Lab Final", layout="wide", page_icon="🧪")

st.markdown("""
<style>
    .stApp {background-color: #0f1116;}
    div[data-testid="stMetricValue"] {color: #00ff41;}
    h1 {color: #00ff41;}
    .stTextInput input {font-size: 20px;}
    .stSelectbox div[data-testid="stMarkdownContainer"] p {font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)


# 1. DATASET
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('DSL-StrongPasswordData.csv')
        return df
    except:
        return None


full_df = load_data()


# 2. LOGGER
class RhythmLogger:
    def __init__(self):
        self.event_queue = []
        self.listener = None
        self.start_listener()

    def start_listener(self):
        if self.listener is None or not self.listener.is_alive():
            self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()

    def clear_queue(self):
        self.event_queue = []

    def _get_key_str(self, key):
        try:
            if hasattr(key, 'vk') and 96 <= key.vk <= 105: return str(key.vk - 96)
            if hasattr(key, 'char') and key.char: return key.char
            return str(key)
        except:
            return str(key)

    def on_press(self, key):
        try:
            t = time.time()
            k = self._get_key_str(key)
            self.event_queue.append({'key': k, 'action': 'press', 'time': t, 'obj': key})
        except:
            pass

    def on_release(self, key):
        try:
            t = time.time()
            k = self._get_key_str(key)
            self.event_queue.append({'key': k, 'action': 'release', 'time': t, 'obj': key})
        except:
            pass


if 'bg_logger' not in st.session_state:
    st.session_state.bg_logger = RhythmLogger()
else:
    st.session_state.bg_logger.start_listener()


# 3. VERİ İŞLEME
def process_data(raw_events):
    enter_times = [e['time'] for e in raw_events if e['obj'] == keyboard.Key.enter and e['action'] == 'release']
    if not enter_times: return None, "Enter Bekleniyor...", []

    last_enter = enter_times[-1]
    events = [e for e in raw_events if last_enter - e['time'] < 30.0 and e['time'] <= last_enter]
    events = sorted(events, key=lambda x: x['time'])

    typed_chars = []
    for ev in events:
        k_val = ev['key']
        k_obj = ev['obj']
        action = ev['action']
        t = ev['time']

        if action == 'press':
            if k_obj == keyboard.Key.backspace:
                if len(typed_chars) > 0: typed_chars.pop()
            elif k_obj in [keyboard.Key.enter, keyboard.Key.shift, keyboard.Key.shift_r]:
                pass
            else:
                if k_val is not None and len(k_val) == 1:
                    typed_chars.append({'char': k_val, 'press': t, 'release': None})
        elif action == 'release':
            if k_val is not None and len(k_val) == 1:
                for item in reversed(typed_chars):
                    if item['char'] == k_val and item['release'] is None:
                        item['release'] = t
                        break

    final_text = "".join([i['char'] for i in typed_chars])
    if final_text != ".tie5Roanl": return None, final_text, []

    features = []
    labels = []

    try:
        dot = typed_chars[0]
        h_dot = (dot['release'] - dot['press']) if dot['release'] else 0.1
        features.append(h_dot)
        labels.append("H(.)")

        prev = typed_chars[0]
        for curr in typed_chars[1:]:
            if curr['release'] is None: curr['release'] = curr['press'] + 0.1
            if prev['release'] is None: prev['release'] = prev['press'] + 0.1

            dd = curr['press'] - prev['press']
            ud = curr['press'] - prev['release']
            h = curr['release'] - curr['press']

            features.extend([dd, ud, h])
            labels.append(f"F({prev['char']}{curr['char']})")  # Flight
            labels.append(f"UD({prev['char']}{curr['char']})")  # UpDown
            labels.append(f"H({curr['char']})")  # Hold

            prev = curr

        features.extend([0.1, 0.1, 0.1])
        labels.extend(["F(lE)", "UD(lE)", "H(E)"])
        return np.array(features).reshape(1, -1), final_text, labels

    except Exception as e:
        return None, str(e), []


# STATE
if 'training_data' not in st.session_state: st.session_state.training_data = []
if 'trained_model' not in st.session_state: st.session_state.trained_model = None
if 'avg_profile' not in st.session_state: st.session_state.avg_profile = None
if 'temp_labels' not in st.session_state: st.session_state.temp_labels = []
if 'feature_labels' not in st.session_state: st.session_state.feature_labels = []
if 'test_result' not in st.session_state: st.session_state.test_result = None

# --- ARAYÜZ ---
st.title("🧪 Biyometrik Güvenlik Laboratuvarı")
st.caption("Kendi profilinizi eğitin, ardından canlı test yapın veya dataset'teki saldırganları simüle edin.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Ayarlar")
    # --- GÜNCELLEME 1: Neural Network seçeneği eklendi ---
    model_choice = st.selectbox("Algoritma:", [
        "Random Forest",
        "SVM",
        "KNN",
        "Neural Network"
    ])

    if st.session_state.avg_profile is not None:
        st.success("Profil Aktif ✅")
        if model_choice == "KNN (Görsel Benzerlik)":
            st.info("ℹ️ KNN seçili: Grafikteki çubuklar ne kadar yakınsa, puan o kadar yüksek olur.")
    else:
        st.warning("⚠️ Önce eğitim yapmalısınız.")

with col2:
    tabs = st.tabs(["🎓 Eğitim", "⚔️ Laboratuvar (Test & Simülasyon)"])

    # --- EĞİTİM ---
    with tabs[0]:
        st.subheader("Profil Oluştur")

        with st.form("train_form", clear_on_submit=True):
            st.markdown("### `.tie5Roanl`")
            inp = st.text_input("Şifreyi yaz:", key="train_inp")
            submitted = st.form_submit_button("Kaydet (Enter)")

            if submitted:
                time.sleep(0.1)
                logger = st.session_state.bg_logger
                real_features, status, labels = process_data(logger.event_queue)
                logger.clear_queue()

                if real_features is not None:
                    st.session_state.training_data.append(real_features)
                    st.session_state.temp_labels = labels
                    st.success("✅ Kaydedildi!")
                else:
                    st.error("Giriş Başarısız")
                    if status: st.caption(f"Durum: {status}")

        st.metric("Veri Sayısı", len(st.session_state.training_data))

        if len(st.session_state.training_data) >= 3:
            if st.button("🚀 Modeli Eğit"):
                # Eğitim Verisi
                X_me = np.vstack(st.session_state.training_data)
                st.session_state.avg_profile = np.mean(X_me, axis=0)
                if st.session_state.temp_labels:
                    st.session_state.feature_labels = st.session_state.temp_labels

                # Dataset (Negatif Örnekler)
                y_me = np.ones(len(X_me))
                if full_df is not None:
                    attackers_pool = full_df.iloc[:, 3:].values
                    n_atk = len(X_me) * 15
                    idx = np.random.choice(attackers_pool.shape[0], n_atk, replace=False)
                    X_ds = attackers_pool[idx]
                    if X_ds.shape[1] > 31: X_ds = X_ds[:, -31:]
                    y_ds = np.zeros(len(X_ds))
                else:
                    st.error("Dataset Yok"); st.stop()

                X_train = np.vstack((X_me, X_ds))
                y_train = np.concatenate((y_me, y_ds))

                # --- GÜNCELLEME 2: Neural Network Mantığı Eklendi ---
                if model_choice == "Random Forest":
                    clf = RandomForestClassifier(n_estimators=100)
                elif model_choice == "SVM (Katı)":
                    clf = make_pipeline(StandardScaler(), SVC(probability=True))
                elif model_choice == "KNN":
                    n_neighbors = min(len(X_me), 3)
                    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors))
                elif model_choice == "Neural Network":
                    clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), max_iter=500))

                clf.fit(X_train, y_train)
                st.session_state.trained_model = clf
                st.success(f"✅ {model_choice} Eğitildi!")

    # --- LABORATUVAR ---
    with tabs[1]:

        t1, t2 = st.columns(2)

        # 1. CANLI GİRİŞ
        with t1:
            st.subheader("👤 Canlı Giriş")
            with st.form("test_form", clear_on_submit=True):
                st.markdown("### `.tie5Roanl`")
                t_inp = st.text_input("Giriş yap:", key="test_inp")
                t_sub = st.form_submit_button("Test Et")

                if t_sub:
                    time.sleep(0.1)
                    logger = st.session_state.bg_logger
                    real_features, status, _ = process_data(logger.event_queue)
                    logger.clear_queue()

                    if real_features is not None:
                        st.session_state.test_result = {
                            "type": "Canlı",
                            "features": real_features,
                            "name": "Sen"
                        }
                    else:
                        st.error("Giriş Başarısız")
                        if status: st.caption(f"Sebep: {status}")

        # 2. SALDIRGAN SİMÜLASYONU
        with t2:
            st.subheader("🤖 Saldırgan Simülasyonu")
            if full_df is not None:
                subjects = full_df['subject'].unique()
                selected_sub = st.selectbox("Saldırgan Seç:", subjects)

                if st.button(f"{selected_sub} Olarak Giriş Yap"):
                    victim_data = full_df[full_df['subject'] == selected_sub].sample(1)
                    feats = victim_data.iloc[:, 3:].values
                    if feats.shape[1] > 31: feats = feats[:, -31:]

                    st.session_state.test_result = {
                        "type": "Simülasyon",
                        "features": feats,
                        "name": f"Saldırgan ({selected_sub})"
                    }
            else:
                st.error("Dataset Yüklenemedi")

        st.divider()

        # --- ORTAK SONUÇ EKRANI ---
        if st.session_state.test_result is not None and st.session_state.trained_model:
            res = st.session_state.test_result
            feats = res['features']

            # Tahmin
            try:
                proba = st.session_state.trained_model.predict_proba(feats)[0][1]
            except:
                proba = st.session_state.trained_model.predict(feats)[0]

            # Sonuç Başlığı
            r1, r2 = st.columns([1, 3])
            with r1:
                st.write(f"### Giriş Yapan: {res['name']}")
                if proba > 0.5:
                    st.success(f"✅ KABUL (%{proba * 100:.1f})")
                else:
                    st.error(f"⛔ RED (%{proba * 100:.1f})")

            with r2:
                if st.session_state.avg_profile is not None and st.session_state.feature_labels:
                    display_limit = 28
                    profile_feats = st.session_state.avg_profile[:display_limit]
                    current_feats = feats.flatten()[:display_limit]
                    labels = st.session_state.feature_labels[:display_limit]

                    chart_df = pd.DataFrame({
                        "Özellik": labels * 2,
                        "Değer (Saniye)": np.concatenate([profile_feats, current_feats]),
                        "Tip": ["Profil (Mavi)"] * len(profile_feats) + ["Şu Anki (Kırmızı)"] * len(current_feats)
                    })

                    st.write("### 📊 Profil Karşılaştırması")

                    # ALTAIR CHART
                    c = alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X('Özellik', sort=None, axis=alt.Axis(labelAngle=-90)),
                        y='Değer (Saniye)',
                        color=alt.Color('Tip', scale={'domain': ['Profil (Mavi)', 'Şu Anki (Kırmızı)'],
                                                      'range': ['#1f77b4', '#d62728']}),
                        xOffset='Tip',
                        tooltip=['Özellik', 'Tip', 'Değer (Saniye)']
                    ).properties(height=350)

                    st.altair_chart(c, width="stretch")

        elif st.session_state.test_result is not None and not st.session_state.trained_model:
            st.warning("⚠️ Lütfen önce eğitim sekmesinden modelinizi eğitin.")