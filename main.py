import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten

# 1. VERİ YÜKLEME VE ÖN İŞLEME (CV Odaklı Kısım)
def load_data(data_dir):
    images = []
    labels = []
    categories = ['normal', 'potholes']
    
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category) # normal -> 0, potholes -> 1 olarak etiketlenecek
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path) 
                img = cv2.resize(img, (128, 128)) # Standart boyut
                
                # CV Dersi için ekstra: Medyan Filtresi ile gürültü azaltma
                img = cv2.medianBlur(img, 3) 
                
                images.append(img)
                labels.append(class_num)
            except Exception as e:
                pass
    return np.array(images), np.array(labels)

X, y = load_data("D:/Desktop/UNI/4.0 Grade/4.2/Vision-Pattern Proje/dataset")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- ALGORİTMA 1: HOG + SVM Başlıyor ---")
# CV KISMI: HOG Öznitelik Çıkarımı
def extract_hog_features(images):
    hog_features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), 
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

# PR KISMI: SVM Sınıflandırma
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_hog, y_train)
svm_predictions = svm_model.predict(X_test_hog)

svm_acc = accuracy_score(y_test, svm_predictions)
print(f"HOG + SVM Doğruluk (Accuracy): {svm_acc * 100:.2f}%")


print("\n--- ALGORİTMA 2: VGG16 (CNN) Başlıyor ---")
# CV + PR KISMI: VGG16 Derin Öğrenme Modeli
# Normalizasyon
X_train_cnn = X_train / 255.0
X_test_cnn = X_test / 255.0

# VGG16 Tabanını Yükleme (Feature Extraction)
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
vgg_base.trainable = False # Önceden eğitilmiş ağırlıkları donduruyoruz

cnn_model = Sequential([
    vgg_base,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid') # Binary classification (Normal vs Anomaly)
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# PR KISMI: Model Eğitimi
cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_data=(X_test_cnn, y_test))

# Test ve Karşılaştırma
loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test)
print(f"VGG16 Doğruluk (Accuracy): {cnn_acc * 100:.2f}%")

print("\n--- SONUÇ KARŞILAŞTIRMASI ---")
print(f"Geleneksel (SVM) Başarımı: {svm_acc * 100:.2f}%")
print(f"Derin Öğrenme (VGG16) Başarımı: {cnn_acc * 100:.2f}%")


import tensorflow as tf
# Eğitilmiş cnn_model'ini TFLite formatına dönüştür
converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)

# İsteğe bağlı: Modeli daha da küçültmek ve hızlandırmak için optimizasyon (Quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT] 

tflite_model = converter.convert()

# Modeli kaydet
with open('vgg16_anomaly_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite modeli başarıyla kaydedildi!")


import cv2
import numpy as np
from keras.applications import VGG16
from keras.models import Model

# --- 1. ÇÖZÜM: DAHA DERİN BİR KATMAN KULLANIYORUZ ---
# block1_conv1 yerine block5_conv3 kullanıyoruz çünkü derin katmanlar nesneleri (çukuru) tanır.
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
layer_name = 'block5_conv3' 
vgg_feature_extractor = Model(inputs=vgg_base.input, outputs=vgg_base.get_layer(layer_name).output)

# --- 2. VİDEO TEST AYARLARI ---
input_video_path = "D:/Desktop/UNI/4.0 Grade/4.2/Vision-Pattern Proje/test_video.mp4"
output_video_path = "D:/Desktop/UNI/4.0 Grade/4.2/Vision-Pattern Proje/outputs/cukur_cerceveli_video.avi"

cap = cv2.VideoCapture(input_video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print("Gelişmiş çerçeveli video işleniyor, lütfen bekleyin...")
kare_sayaci = 0

while cap.isOpened():
    ret, frame = cap.read() 
    if not ret:
        break 
        
    kare_sayaci += 1
    
    # Ön İşleme
    img = cv2.resize(frame, (128, 128)) 
    img_blur = cv2.medianBlur(img, 3)        
    img_norm = img_blur / 255.0                   
    img_expanded = np.expand_dims(img_norm, axis=0) 
    
    # VGG16 Modeli ile Çukur Tahmini
    prediction = cnn_model.predict(img_expanded, verbose=0)[0][0]
    
    if prediction > 0.5: 
        
        # Feature Map al (Artık 5. Bloktan alıyoruz, daha akıllı!)
        feature_maps = vgg_feature_extractor.predict(img_expanded, verbose=0)
        activation_map = np.mean(feature_maps[0], axis=-1)
        
        # Haritayı orijinal boyuta getir
        activation_map = (activation_map - np.min(activation_map)) / (np.max(activation_map) - np.min(activation_map) + 1e-5)
        activation_map = (activation_map * 255).astype(np.uint8)
        activation_map_resized = cv2.resize(activation_map, (width, height))
        
        # ÇÖZÜM: Odak haritasını yumuşat (Gaussian Blur) ki kare kare pikseller çıksın
        activation_map_resized = cv2.GaussianBlur(activation_map_resized, (15, 15), 0)
        
        # ÇÖZÜM: Dinamik Eşikleme (Sadece en çok odaklanılan %25'lik parlak alanı al)
        max_val = np.max(activation_map_resized)
        dynamic_threshold = max_val * 0.75 
        _, thresh = cv2.threshold(activation_map_resized, dynamic_threshold, 255, cv2.THRESH_BINARY)
        
        # Sınırları (Contours) bul
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # ÇÖZÜM: Aşırı küçük tozları ve tüm ekranı kaplayan devasa hataları yoksay
            if 800 < area < (width * height * 0.4): 
                x, y, w, h = cv2.boundingRect(cnt)
                
                # ÇÖZÜM: Çerçeve x=0 veya y=0'a (tam köşeye) sıfıra sıfır yapışıyorsa onu çizme (Sınır Yanılgısı Filtresi)
                if x > 5 and y > 5:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    cv2.putText(frame, f"Cukur ({(prediction*100):.0f}%)", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.putText(frame, "DURUM: ANOMALI", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    else:
        cv2.putText(frame, "DURUM: YOL TEMIZ", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video başarıyla işlendi! Köşe hataları giderildi ve akıllı çerçeveler çizildi.")
