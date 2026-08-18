# CARAS — Classification Accuracy and Regression Assessment Suite

CARAS, iki raster harita (ör. sınıflandırılmış vs. referans/ground truth) arasında kapsamlı doğrulama ve regresyon analizi yapan genel amaçlı bir QGIS eklentisidir.

CARAS is a general-purpose QGIS plugin for performing comprehensive accuracy and regression assessment between two raster maps (e.g., a classified map vs. a reference/ground truth map).

## Özellikler / Features

- **Örnekleme yöntemleri / Sampling methods**: Rastgele, katmanlı (stratified), sistematik veya önceden hazırlanmış CSV noktaları
- **Klasörden raster seçimi**: Katman listesindeki rasterlerin yanı sıra "…" butonuyla doğrudan diskten raster dosyası açılabilir
- **Sınıf eşleştirme arayüzü**: Referans ve sınıflandırılmış haritadaki piksel değerlerini karşılaştırılabilir kategorilere eşleme
- **Doğrulama metrikleri**: Overall Accuracy, Cohen's Kappa, F1-Score (macro & weighted), Precision, Recall, Confusion Matrix, Producer's & User's Accuracy
- **Regresyon istatistikleri**: R², RMSE, MAE, Bias (hem ham piksel hem kategori değerleri için)
- **Rapor çıktıları**: TXT, JSON, HTML
- **Nokta katmanı dışa aktarımı**: Shapefile

## Gereksinimler / Requirements

- QGIS 3.x veya 4.0
- Python paketleri: `numpy`, `scikit-learn` (QGIS'in kendi Python ortamına kurulmalı)

```
pip install numpy scikit-learn
```

## Kurulum / Installation

1. Bu depoyu indirin veya klonlayın
2. `caras` klasörünü QGIS eklenti klasörüne kopyalayın:
   - Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\` (veya `QGIS4\...`)
   - Linux/macOS: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
3. QGIS'i başlatın, **Eklentiler → Eklentileri Yönet ve Yükle** üzerinden CARAS'ı etkinleştirin

## Kullanım / Usage

1. Araç çubuğundaki CARAS ikonuna tıklayın
2. Referans ve sınıflandırılmış haritayı katman listesinden seçin ya da "…" ile diskten açın
3. Örnekleme yöntemini ve nokta sayısını belirleyin (veya CSV dosyası seçin)
4. **CARAS Analizi Başlat** ile analizi çalıştırın, açılan pencerede sınıf eşleştirmesini yapın
5. Sonuçları inceleyin, isterseniz raporu (TXT/JSON/HTML) veya doğrulama noktalarını (Shapefile) kaydedin

CSV ile önceden belirlenmiş nokta kullanımı için bkz. [CSV_NOKTA_KULLANIMI.md](CSV_NOKTA_KULLANIMI.md).

## Lisans / License

Yazar / Author: Ömer K. ÖRÜCÜ — omerorucu@sdu.edu.tr

Bu eklenti DeepSeek AI ve Claude AI (Anthropic) yardımıyla geliştirilmiştir.
This plugin was developed with the assistance of DeepSeek AI and Claude AI (Anthropic).
