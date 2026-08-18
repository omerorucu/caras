# CARAS — CSV ile Önceden Belirlenmiş Noktaları Kullanma
# CARAS — Using Predefined Points from CSV

## 📋 Genel Bakış / Overview

Bu özellik, arazi çalışmasından veya diğer kaynaklardan elde edilmiş önceden belirlenmiş doğrulama noktalarını CARAS ile kullanmanıza olanak tanır.

This feature allows you to use predefined validation points obtained from field work or other sources with CARAS.

---

## 📁 CSV Dosya Formatı / CSV File Format

### Gerekli Sütunlar / Required Columns

CSV dosyanız **mutlaka** şu 4 sütunu içermelidir (başlık satırı dahil):

Your CSV file **must** contain these 4 columns (including header row):

```csv
id,x,y,reference_value
```

| Sütun / Column | Açıklama / Description | Örnek / Example |
|----------------|------------------------|--------------------|
| `id` | Nokta kimliği (benzersiz) / Point identifier (unique) | P001, Site_A, 1 |
| `x` | Boylam (WGS 84) / Longitude (WGS 84) | 30.5234 |
| `y` | Enlem (WGS 84) / Latitude (WGS 84) | 37.8765 |
| `reference_value` | Referans sınıf değeri (tam/ondalıklı) / Reference class value (integer/float) | 1, 2.5, 0.234 |

### 📌 reference_value Formatı

**reference_value** hem **tam sayı** hem de **ondalıklı sayı** olabilir:

#### Tam Sayı Örneği / Integer Example
```csv
id,x,y,reference_value
P001,30.5234,37.8765,1
P002,30.5456,37.8901,2
P003,30.5678,37.9023,3
```

#### Ondalıklı Sayı Örneği / Float Example
```csv
id,x,y,reference_value
Site_A,30.5234,37.8765,0.234
Site_B,30.5456,37.8901,2.567
Site_C,30.5678,37.9023,4.123
```

#### Karışık Format / Mixed Format
```csv
id,x,y,reference_value
P001,30.5234,37.8765,1.0
P002,30.5456,37.8901,2.5
P003,30.5678,37.9023,3
P004,30.5890,37.9145,4.75
```

---

## ✅ Örnek CSV Dosyası / Sample CSV File

### Tam Sayı Değerler / Integer Values
```csv
id,x,y,reference_value
P001,30.5234,37.8765,1
P002,30.5456,37.8901,2
P003,30.5678,37.9023,1
P004,30.5890,37.9145,3
P005,30.6012,37.9267,2
```

### Ondalıklı Değerler / Float Values
```csv
id,x,y,reference_value
Site_A,30.5234,37.8765,0.234
Site_B,30.5456,37.8901,0.567
Site_C,30.5678,37.9023,0.789
Site_D,30.5890,37.9145,0.456
Site_E,30.6012,37.9267,0.891
```

---

## 🌍 Koordinat Sistemi / Coordinate System

**ÖNEMLİ / IMPORTANT:** Koordinatlar **WGS 84 (EPSG:4326)** sisteminde olmalıdır!

Coordinates **must be** in **WGS 84 (EPSG:4326)** system!

- **X (Boylam/Longitude)**: -180 ile +180 arası / between -180 and +180
- **Y (Enlem/Latitude)**: -90 ile +90 arası / between -90 and +90
- **Ondalık derece formatı** kullanın / Use **decimal degrees** format

CARAS, WGS 84 koordinatlarını otomatik olarak projenizin koordinat sistemine dönüştürür.
CARAS automatically transforms WGS 84 coordinates to your project's coordinate system.

---

## 📍 Kullanım Adımları / Usage Steps

### 1. CSV Dosyasını Hazırlayın / Prepare CSV File

```csv
id,x,y,reference_value
Point_1,30.123456,38.654321,1
Point_2,30.234567,38.765432,2
Point_3,30.345678,38.876543,3
```

### 2. CARAS'ta CSV Seçeneğini Seçin / Select CSV Option in CARAS

1. **Örnekleme Yöntemi** / **Sampling Method** → "CSV Dosyasından / From CSV" seçin
2. **Gözat** / **Browse** butonuna tıklayın
3. CSV dosyanızı seçin / Select your CSV file

### 3. Analizi Çalıştırın / Run Analysis

- CARAS otomatik olarak:
  - CSV'den noktaları yükler / Loads points from CSV
  - Koordinatları dönüştürür / Transforms coordinates
  - Sınıflandırılmış haritadan değerleri okur / Reads values from classified map
  - Referans değerleri CSV'den alır / Takes reference values from CSV

---

## 📊 Sonuçlarda CSV Bilgileri / CSV Information in Results

Analiz tamamlandığında, shapefile çıktısında şu ek bilgiler yer alır:

| Alan / Field | Açıklama / Description |
|--------------|------------------------|
| `csv_id` | CSV'deki orijinal nokta ID'si / Original point ID from CSV |
| `ref_value` | CSV'den gelen referans değeri / Reference value from CSV |
| `class_val` | Sınıflandırılmış haritadan okunan değer / Value read from classified map |
| `match` | Eşleşme durumu (Yes/No) / Match status (Yes/No) |

---

## 🆚 CSV vs Random/Stratified Sampling

| Özellik / Feature | CSV | Random | Stratified |
|-------------------|-----|--------|------------|
| **Nokta Seçimi** / **Point Selection** | Kullanıcı tanımlı / User-defined | Otomatik rastgele / Auto random | Otomatik grid / Auto grid |
| **Referans Değer** / **Reference Value** | CSV'den / From CSV | Haritadan / From map | Haritadan / From map |
| **Arazi Çalışması** / **Field Work** | Uygun / Suitable | Uygun değil / Not suitable | Uygun değil / Not suitable |
| **Tekrarlanabilirlik** / **Reproducibility** | Yüksek / High | Düşük / Low | Orta / Medium |
| **Esneklik** / **Flexibility** | Yüksek / High | Düşük / Low | Orta / Medium |

---

## ⚠️ Önemli Notlar / Important Notes

1. **Koordinat Sistemi**: Koordinatlar **mutlaka WGS 84** olmalı!
   **Coordinate System**: Coordinates **must be WGS 84**!

2. **Karakter Kodlaması**: CSV dosyanız **UTF-8** kodlamasında olmalı
   **Character Encoding**: CSV file should be in **UTF-8** encoding

3. **Ondalık Ayırıcı**: Nokta (.) kullanın, virgül (,) değil
   **Decimal Separator**: Use dot (.), not comma (,)

4. **Boş Değerler**: Boş satır veya eksik değer olmamalı
   **Empty Values**: No empty rows or missing values

5. **Minimum Nokta Sayısı**: En az **30 nokta** önerilir / At least **30 points** recommended

---

## 🔧 Sorun Giderme / Troubleshooting

**"Koordinatlar harita sınırları dışında"**
- Koordinatların WGS 84 formatında olduğundan emin olun
- X ve Y sütunlarını kontrol edin (karışıklık olabilir)

**"Geçersiz referans değerleri"**
- reference_value sütununun sayısal değerler içerdiğinden emin olun
- Ondalık ayırıcı olarak nokta (.) kullanın

**"CSV dosyası okunamıyor"**
- Dosya UTF-8 kodlamasında mı?
- Virgülle ayrılmış mı?
- Excel'de açıp CSV olarak yeniden kaydedin

---

**CARAS ile başarılı analizler! / Successful analyses with CARAS!** 🚀
