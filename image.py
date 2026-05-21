import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- 1. AYARLAR VE STİL ---
# Seaborn stilini aktif et (Daha modern görünüm için)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300  # Yüksek çözünürlük
plt.rcParams['font.family'] = 'sans-serif'


def veri_yukle():
    try:
        df = pd.read_csv('DSL-StrongPasswordData.csv')
        return df
    except:
        print("HATA: CSV dosyası bulunamadı.")
        return None


# --- 2. VERİ HAZIRLIĞI ---
df = veri_yukle()
if df is not None:
    target_user = 's002'
    X = df.iloc[:, 3:].values
    y = (df['subject'] == target_user).astype(int)
    feature_names = df.columns[3:]  # Özellik isimlerini al

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- 3. MODELLERİN EĞİTİMİ VE METRİK HESABI ---
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC(random_state=42)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Neural Network": make_pipeline(StandardScaler(),
                                        MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
    }

    results = []

    # Random Forest'ı ayrıca tutalım (Confusion Matrix ve Feature Importance için)
    rf_model = models["Random Forest"]
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    # Tüm modelleri döngüye sok
    for name, model in models.items():
        if name != "Random Forest":  # RF'i zaten yukarıda eğittik
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "Algoritma": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

    results_df = pd.DataFrame(results)

    # --- GÖRSEL 1: SONUÇ TABLOSU (RESİM OLARAK) ---
    print("1. Tablo Oluşturuluyor...")
    fig, ax = plt.subplots(figsize=(10, 3))  # Boyut
    ax.axis('tight')
    ax.axis('off')

    # Tablo verilerini formatla (% sembolü ekle)
    fmt_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        fmt_df[col] = fmt_df[col].apply(lambda x: f"%{x * 100:.2f}")

    table = ax.table(cellText=fmt_df.values,
                     colLabels=fmt_df.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)  # Hücre genişliği ve yüksekliği

    # Başlık satırını renklendir
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')  # Koyu mavi başlık
        elif row % 2 == 0:
            cell.set_facecolor('#f2f2f2')  # Satırları gri yap

    plt.title("Algoritma Performans Karşılaştırması", fontsize=14, pad=10, weight='bold')
    plt.tight_layout()
    plt.savefig('Rapor_Tablo_Performans.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- GÖRSEL 2: CONFUSION MATRIX (Random Forest) ---
    print("2. Confusion Matrix Oluşturuluyor...")
    plt.figure(figsize=(6, 5))

    # Renkli Isı Haritası
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Saldırgan (Tahmin)', 'Gerçek (Tahmin)'],
                yticklabels=['Saldırgan (Gerçek)', 'Gerçek (Gerçek)'])

    plt.title('Random Forest - Hata Matrisi', fontsize=12, weight='bold')
    plt.ylabel('Gerçek Durum')
    plt.xlabel('Model Tahmini')
    plt.tight_layout()
    plt.savefig('Rapor_Grafik_ConfusionMatrix.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- GÖRSEL 3: FEATURE IMPORTANCE (Öznitelik Önemi) ---
    print("3. Feature Importance Grafiği Oluşturuluyor...")
    # En önemli 10 özelliği al
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")

    plt.title("Model Kararında En Etkili 10 Tuş Özelliği", fontsize=12, weight='bold')
    plt.xlabel("Önem Derecesi")
    plt.ylabel("Özellik Adı")
    plt.tight_layout()
    plt.savefig('Rapor_Grafik_FeatureImportance.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("\n✅ Harika! 3 adet yüksek kaliteli görsel oluşturuldu:")
    print("   1. Rapor_Tablo_Performans.png (Tablo yerine bunu koy)")
    print("   2. Rapor_Grafik_ConfusionMatrix.png (RF'in başarısını gösterir)")
    print("   3. Rapor_Grafik_FeatureImportance.png (Hold Time kanıtı)")

else:
    print("Veri yüklenemediği için işlem iptal edildi.")