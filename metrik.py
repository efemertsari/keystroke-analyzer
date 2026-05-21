import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings

# Gereksiz uyarıları gizle (Temiz çıktı için)
warnings.filterwarnings('ignore')

print("⏳ Veri seti yükleniyor ve modeller eğitiliyor, lütfen bekleyin...\n")

# 1. Veri Setini Yükle
try:
    df = pd.read_csv('DSL-StrongPasswordData.csv')
except FileNotFoundError:
    print("HATA: 'DSL-StrongPasswordData.csv' dosyası bulunamadı!")
    exit()

# 2. Veri Hazırlığı (Binary Classification)
# Hedef kullanıcı 's002' olsun (Datasetin başındaki kişi)
target_user = 's002'

X = df.iloc[:, 3:].values  # Özellikler (Süreler)
y = (df['subject'] == target_user).astype(int)  # 1: Gerçek Kullanıcı, 0: Saldırgan

# Eğitim ve Test setine ayır (%70 Eğitim, %30 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Modellerin Tanımlanması
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Neural Network": make_pipeline(StandardScaler(),
                                    MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
}

# 4. Tablo Başlığı
print(f"{'ALGORİTMA':<20} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
print("-" * 75)

# 5. Eğitim ve Test Döngüsü
for name, clf in models.items():
    # Eğit
    clf.fit(X_train, y_train)

    # Tahmin Et
    y_pred = clf.predict(X_test)

    # Metrikleri Hesapla
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Ekrana Yazdır (Tablo Formatında)
    print(f"{name:<20} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")

print("-" * 75)
print("\n✅ İşlem tamamlandı! Bu değerleri raporunun tablosuna yazabilirsin.")