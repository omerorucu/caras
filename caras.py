# -*- coding: utf-8 -*-
"""
CARAS - Classification Accuracy and Regression Assessment Suite
İki raster harita arasında doğrulama analizi yapan genel amaçlı plugin
QGIS 4.0 uyumlu versiyon - PyQt6 / QGIS 4.x API güncellemeleri uygulandı
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QFont
from qgis.PyQt.QtWidgets import (QAction, QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QSpinBox, QPushButton, QComboBox, QTextEdit, QGroupBox, QFileDialog, QMessageBox, 
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QRadioButton,
    QButtonGroup, QWidget, QScrollArea, QLineEdit, QApplication)
from qgis.core import (QgsProject, QgsVectorLayer, QgsRasterLayer, QgsField, 
                       QgsFeature, QgsGeometry, QgsPointXY,
                       QgsVectorFileWriter, QgsVectorFileWriterTask,
                       QgsCoordinateTransformContext,
                       QgsWkbTypes, QgsApplication)
from qgis.utils import iface

# QGIS 4.0 / PyQt6 uyumluluk: QVariant kaldırıldı, Python türleri kullanılır
try:
    from qgis.PyQt.QtCore import QVariant
    _USE_QVARIANT = True
except ImportError:
    _USE_QVARIANT = False
import numpy as np
import random
try:
    from sklearn.metrics import (cohen_kappa_score, accuracy_score, confusion_matrix,
        classification_report, f1_score, precision_score, recall_score,
        mean_squared_error, mean_absolute_error, r2_score)
except ImportError as e:
    raise ImportError(
        "CARAS eklentisi icin 'scikit-learn' kutuphanesi gereklidir.\n"
        "CARAS plugin requires the 'scikit-learn' library.\n\n"
        "Kurulum / Install (OSGeo4W Shell veya QGIS Python Console):\n"
        "  pip install scikit-learn\n\n"
        f"Orijinal hata / Original error: {e}"
    ) from e
import os
import json
from datetime import datetime


class ClassMappingDialog(QDialog):
    """Sınıf eşleştirme için dialog"""
    def __init__(self, reference_values, classified_values, parent=None):
        super(ClassMappingDialog, self).__init__(parent)
        self.setWindowTitle("CARAS - Sınıf Eşleştirme / Class Mapping")
        self.setMinimumWidth(900)
        self.setMinimumHeight(700)
        
        self.reference_unique = sorted(list(set(reference_values)))
        self.classified_unique = sorted(list(set(classified_values)))
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Başlık ve açıklama
        title = QLabel("Sınıf Eşleştirme Ayarları")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        info = QLabel(
            "Her iki haritadaki sınıf değerlerini karşılaştırılabilir kategorilere atayın.\n"
            "Aynı anlamı taşıyan sınıflar aynı kategori numarasına sahip olmalıdır.\n"
            "Kategoriler: 1, 2, 3, 4, 5... şeklinde numaralandırılır."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; padding: 10px; background: #f0f0f0; border-radius: 5px;")
        layout.addWidget(info)
        
        # Scroll area için container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QHBoxLayout()
        
        # Referans harita eşleştirme
        reference_group = QGroupBox("Referans Haritası (Ground Truth)")
        reference_layout = QVBoxLayout()
        
        ref_info = QLabel("Bu harita gerçeği temsil eden referans haritasıdır.")
        ref_info.setStyleSheet("color: #2c3e50; font-style: italic;")
        reference_layout.addWidget(ref_info)
        
        self.reference_table = QTableWidget()
        self.reference_table.setColumnCount(3)
        self.reference_table.setHorizontalHeaderLabels(["Piksel Değeri", "Sınıf Adı", "Kategori"])
        self.reference_table.setRowCount(len(self.reference_unique))
        
        for i, val in enumerate(self.reference_unique):
            if isinstance(val, float):
                if val == int(val):
                    display_val = str(int(val))
                else:
                    display_val = f"{val:.4f}"
            else:
                display_val = str(val)
                
            item = QTableWidgetItem(display_val)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.reference_table.setItem(i, 0, item)
            
            if isinstance(val, float):
                if val == int(val):
                    default_name = f"Sınıf_{int(val)}"
                else:
                    default_name = f"Sınıf_{val:.2f}"
            else:
                default_name = f"Sınıf_{val}"
                
            name_edit = QLineEdit(default_name)
            self.reference_table.setCellWidget(i, 1, name_edit)
            
            category_spin = QSpinBox()
            category_spin.setMinimum(1)
            category_spin.setMaximum(100)
            category_spin.setValue(i + 1)
            self.reference_table.setCellWidget(i, 2, category_spin)
            
        self.reference_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        reference_layout.addWidget(self.reference_table)
        reference_group.setLayout(reference_layout)
        scroll_layout.addWidget(reference_group)
        
        # Sınıflandırılmış harita eşleştirme
        classified_group = QGroupBox("Sınıflandırılmış Harita (Classification)")
        classified_layout = QVBoxLayout()
        
        class_info = QLabel("Bu harita doğruluğu değerlendirilen sınıflandırılmış haritadır.")
        class_info.setStyleSheet("color: #2c3e50; font-style: italic;")
        classified_layout.addWidget(class_info)
        
        self.classified_table = QTableWidget()
        self.classified_table.setColumnCount(3)
        self.classified_table.setHorizontalHeaderLabels(["Piksel Değeri", "Sınıf Adı", "Kategori"])
        self.classified_table.setRowCount(len(self.classified_unique))
        
        for i, val in enumerate(self.classified_unique):
            if isinstance(val, float):
                if val == int(val):
                    display_val = str(int(val))
                else:
                    display_val = f"{val:.4f}"
            else:
                display_val = str(val)
                
            item = QTableWidgetItem(display_val)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.classified_table.setItem(i, 0, item)
            
            if isinstance(val, float):
                if val == int(val):
                    default_name = f"Sınıf_{int(val)}"
                else:
                    default_name = f"Sınıf_{val:.2f}"
            else:
                default_name = f"Sınıf_{val}"
                
            name_edit = QLineEdit(default_name)
            self.classified_table.setCellWidget(i, 1, name_edit)
            
            category_spin = QSpinBox()
            category_spin.setMinimum(1)
            category_spin.setMaximum(100)
            category_spin.setValue(i + 1)
            self.classified_table.setCellWidget(i, 2, category_spin)
            
        self.classified_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        classified_layout.addWidget(self.classified_table)
        classified_group.setLayout(classified_layout)
        scroll_layout.addWidget(classified_group)
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Hızlı eşleştirme butonları
        quick_layout = QHBoxLayout()
        quick_label = QLabel("Hızlı Eşleştirme:")
        quick_layout.addWidget(quick_label)
        
        auto_button = QPushButton("Otomatik Eşleştir (Sıralı)")
        auto_button.setToolTip("Her iki haritanın değerlerini küçükten büyüğe sıralayarak eşleştirir")
        auto_button.clicked.connect(self.auto_map_sequential)
        quick_layout.addWidget(auto_button)
        
        identical_button = QPushButton("Aynı Değerler (1:1)")
        identical_button.setToolTip("Aynı piksel değerlerini aynı kategoriye atar")
        identical_button.clicked.connect(self.auto_map_identical)
        quick_layout.addWidget(identical_button)
        
        quick_layout.addStretch()
        layout.addLayout(quick_layout)
        
        # Butonlar
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("Tamam")
        ok_button.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-weight: bold; padding: 8px; }")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("İptal")
        cancel_button.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; font-weight: bold; padding: 8px; }")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def auto_map_sequential(self):
        """Sıralı otomatik eşleştirme"""
        for i in range(self.reference_table.rowCount()):
            category_spin = self.reference_table.cellWidget(i, 2)
            category_spin.setValue(i + 1)
            
        for i in range(self.classified_table.rowCount()):
            category_spin = self.classified_table.cellWidget(i, 2)
            category_spin.setValue(i + 1)
            
    def auto_map_identical(self):
        """Aynı değerleri eşleştir"""
        value_to_category = {}
        category_counter = 1
        
        for i in range(self.reference_table.rowCount()):
            val = self.reference_unique[i]
            if val not in value_to_category:
                value_to_category[val] = category_counter
                category_counter += 1
            category_spin = self.reference_table.cellWidget(i, 2)
            category_spin.setValue(value_to_category[val])
            
        for i in range(self.classified_table.rowCount()):
            val = self.classified_unique[i]
            if val not in value_to_category:
                value_to_category[val] = category_counter
                category_counter += 1
            category_spin = self.classified_table.cellWidget(i, 2)
            category_spin.setValue(value_to_category[val])
            
    def get_mappings(self):
        """Eşleştirme bilgilerini al"""
        reference_mapping = {}
        reference_names = {}
        
        for i in range(self.reference_table.rowCount()):
            val = self.reference_unique[i]
            name_edit = self.reference_table.cellWidget(i, 1)
            category_spin = self.reference_table.cellWidget(i, 2)
            
            category = category_spin.value()
            reference_mapping[val] = category
            reference_names[category] = name_edit.text()
            
        classified_mapping = {}
        classified_names = {}
        
        for i in range(self.classified_table.rowCount()):
            val = self.classified_unique[i]
            name_edit = self.classified_table.cellWidget(i, 1)
            category_spin = self.classified_table.cellWidget(i, 2)
            
            category = category_spin.value()
            classified_mapping[val] = category
            if category not in classified_names:
                classified_names[category] = name_edit.text()
            
        final_names = {}
        all_categories = set(list(reference_names.keys()) + list(classified_names.keys()))
        
        for cat in all_categories:
            if cat in reference_names:
                final_names[cat] = reference_names[cat]
            else:
                final_names[cat] = classified_names[cat]
        
        return reference_mapping, classified_mapping, final_names


class CARASDialog(QDialog):
    """CARAS Ana dialog penceresi"""
    def __init__(self, parent=None):
        super(CARASDialog, self).__init__(parent)
        self.setWindowTitle("CARAS - Classification Accuracy and Regression Assessment Suite")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(800)
        
        self.sampled_points = None
        self.validation_results = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Arayüzü oluştur"""
        main_layout = QVBoxLayout()
        
        # Başlık
        title = QLabel("🌍 CARAS — Classification Accuracy and Regression Assessment Suite")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Açıklama
        description = QLabel(
            "İki raster harita arasında kapsamlı doğrulama ve regresyon analizi yapar.\n"
            "Performs comprehensive accuracy and regression assessment between two raster maps."
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setStyleSheet("color: #555; padding: 10px;")
        main_layout.addWidget(description)
        
        # Harita seçimi
        map_group = QGroupBox("1. Harita Seçimi / Map Selection")
        map_layout = QVBoxLayout()
        
        ref_layout = QHBoxLayout()
        ref_label = QLabel("Referans Harita / Reference Map:")
        ref_label.setMinimumWidth(250)
        ref_label.setStyleSheet("font-weight: bold;")
        self.reference_combo = QComboBox()
        self.reference_combo.setMinimumWidth(400)
        ref_layout.addWidget(ref_label)
        ref_layout.addWidget(self.reference_combo)
        ref_layout.addStretch()
        map_layout.addLayout(ref_layout)
        
        ref_info = QLabel("↪ Gerçek arazi durumunu gösteren harita (ground truth)")
        ref_info.setStyleSheet("color: #7f8c8d; font-size: 10pt; margin-left: 20px;")
        map_layout.addWidget(ref_info)
        
        class_layout = QHBoxLayout()
        class_label = QLabel("Sınıflandırılmış Harita / Classified Map:")
        class_label.setMinimumWidth(250)
        class_label.setStyleSheet("font-weight: bold;")
        self.classified_combo = QComboBox()
        self.classified_combo.setMinimumWidth(400)
        class_layout.addWidget(class_label)
        class_layout.addWidget(self.classified_combo)
        class_layout.addStretch()
        map_layout.addLayout(class_layout)
        
        class_info = QLabel("↪ Doğruluğu değerlendirilecek sınıflandırılmış harita")
        class_info.setStyleSheet("color: #7f8c8d; font-size: 10pt; margin-left: 20px;")
        map_layout.addWidget(class_info)
        
        map_group.setLayout(map_layout)
        main_layout.addWidget(map_group)
        
        # Örnekleme ayarları
        sampling_group = QGroupBox("2. Örnekleme Ayarları / Sampling Settings")
        sampling_layout = QVBoxLayout()
        
        method_layout = QHBoxLayout()
        method_label = QLabel("Örnekleme Metodu / Method:")
        method_label.setMinimumWidth(250)
        self.method_group = QButtonGroup()
        
        self.random_radio = QRadioButton("Rastgele / Random")
        self.random_radio.setChecked(True)
        self.stratified_radio = QRadioButton("Katmanlı / Stratified")
        self.systematic_radio = QRadioButton("Sistematik / Systematic")
        self.csv_radio = QRadioButton("CSV Dosyası / CSV File")
        
        self.method_group.addButton(self.random_radio, 1)
        self.method_group.addButton(self.stratified_radio, 2)
        self.method_group.addButton(self.systematic_radio, 3)
        self.method_group.addButton(self.csv_radio, 4)
        
        self.random_radio.toggled.connect(self.on_sampling_method_changed)
        self.csv_radio.toggled.connect(self.on_sampling_method_changed)
        
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.random_radio)
        method_layout.addWidget(self.stratified_radio)
        method_layout.addWidget(self.systematic_radio)
        method_layout.addWidget(self.csv_radio)
        method_layout.addStretch()
        sampling_layout.addLayout(method_layout)
        
        self.csv_widget = QWidget()
        csv_layout = QHBoxLayout()
        csv_layout.setContentsMargins(250, 0, 0, 0)
        
        csv_info = QLabel("CSV Formatı: id, x, y, reference_value")
        csv_info.setStyleSheet("color: #7f8c8d; font-size: 9pt;")
        csv_layout.addWidget(csv_info)
        
        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setPlaceholderText("CSV dosya yolu / CSV file path...")
        self.csv_path_edit.setMinimumWidth(300)
        csv_layout.addWidget(self.csv_path_edit)
        
        csv_browse_button = QPushButton("📁 Gözat / Browse")
        csv_browse_button.clicked.connect(self.browse_csv_file)
        csv_layout.addWidget(csv_browse_button)
        
        csv_layout.addStretch()
        self.csv_widget.setLayout(csv_layout)
        self.csv_widget.setVisible(False)
        sampling_layout.addWidget(self.csv_widget)
        
        points_layout = QHBoxLayout()
        points_label = QLabel("Nokta Sayısı / Number of Points:")
        points_label.setMinimumWidth(250)
        self.points_spin = QSpinBox()
        self.points_spin.setMinimum(30)
        self.points_spin.setMaximum(100000)
        self.points_spin.setValue(500)
        self.points_spin.setSingleStep(50)
        points_layout.addWidget(points_label)
        points_layout.addWidget(self.points_spin)
        points_layout.addStretch()
        sampling_layout.addLayout(points_layout)
        
        sampling_group.setLayout(sampling_layout)
        main_layout.addWidget(sampling_group)
        
        # Çalıştır butonu
        button_layout = QHBoxLayout()
        
        self.validate_button = QPushButton("🔍 CARAS Analizi Başlat / Run CARAS Analysis")
        self.validate_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.validate_button.clicked.connect(self.run_validation)
        button_layout.addWidget(self.validate_button)
        
        main_layout.addLayout(button_layout)
        
        # İlerleme çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Sonuç alanı
        results_group = QGroupBox("3. Sonuçlar / Results")
        results_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(300)
        self.result_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 2px solid #34495e;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        results_layout.addWidget(self.result_text)
        
        action_layout = QHBoxLayout()
        
        self.export_button = QPushButton("💾 Raporu Kaydet / Save Report")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        action_layout.addWidget(self.export_button)
        
        self.save_points_button = QPushButton("📍 Noktaları Kaydet / Save Points")
        self.save_points_button.setEnabled(False)
        self.save_points_button.clicked.connect(self.save_validation_points)
        action_layout.addWidget(self.save_points_button)
        
        action_layout.addStretch()
        results_layout.addLayout(action_layout)
        
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        self.setLayout(main_layout)
        
    def on_sampling_method_changed(self):
        """Örnekleme metoduna göre UI'yi ayarla"""
        is_csv = self.csv_radio.isChecked()
        self.csv_widget.setVisible(is_csv)
        self.points_spin.setEnabled(not is_csv)
        self.reference_combo.setEnabled(not is_csv)
        
    def browse_csv_file(self):
        """CSV dosyası seç"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "CSV Dosyası Seçin / Select CSV File", 
            "", 
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            self.csv_path_edit.setText(file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    second_line = f.readline().strip()
                    
                    headers = [h.strip().lower() for h in first_line.split(',')]
                    required = ['id', 'x', 'y', 'reference_value']
                    
                    if not all(req in headers for req in required):
                        QMessageBox.warning(self, "Uyarı / Warning",
                            f"CSV dosyası gerekli sütunları içermiyor!\n"
                            f"CSV file doesn't contain required columns!\n\n"
                            f"Gerekli / Required: id, x, y, reference_value\n"
                            f"Bulunan / Found: {', '.join(headers)}")
                        self.csv_path_edit.clear()
                        return
                    
                    if second_line:
                        test_data = second_line.split(',')
                        if len(test_data) < 4:
                            QMessageBox.warning(self, "Uyarı / Warning",
                                "CSV formatı hatalı! / Invalid CSV format!\n"
                                "Her satır en az 4 sütun içermelidir / Each row must have at least 4 columns")
                            self.csv_path_edit.clear()
                            return
                            
                QMessageBox.information(self, "Başarılı / Success",
                    "✓ CSV dosyası başarıyla yüklendi!\n"
                    "✓ CSV file loaded successfully!")
                    
            except Exception as e:
                QMessageBox.critical(self, "Hata / Error",
                    f"CSV dosyası okunamadı / Cannot read CSV file:\n{str(e)}")
                self.csv_path_edit.clear()
    
    def load_points_from_csv(self, csv_path, raster_layer):
        """CSV dosyasından noktaları yükle (raster_layer: koordinat/piksel dönüşümü
        için kullanılan sınıflandırılmış harita, referans değerler ise CSV'den gelir)"""
        from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform

        points = []
        reference_values_from_csv = []
        point_ids = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                header = f.readline().strip().split(',')
                headers = [h.strip().lower() for h in header]

                id_idx = headers.index('id')
                x_idx = headers.index('x')
                y_idx = headers.index('y')
                ref_val_idx = headers.index('reference_value')

                wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
                layer_crs = raster_layer.crs()

                needs_transform = (wgs84.authid() != layer_crs.authid())
                if needs_transform:
                    transform = QgsCoordinateTransform(wgs84, layer_crs, QgsProject.instance())

                ref_extent = raster_layer.extent()

                for line_num, line in enumerate(f, start=2):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(',')
                    if len(parts) < 4:
                        continue

                    try:
                        point_id = parts[id_idx].strip()
                        x = float(parts[x_idx].strip())
                        y = float(parts[y_idx].strip())

                        ref_val_str = parts[ref_val_idx].strip()
                        ref_val = float(ref_val_str)

                        point_geom = QgsPointXY(x, y)

                        if needs_transform:
                            point_geom = transform.transform(point_geom)

                        pixel_x = int((point_geom.x() - ref_extent.xMinimum()) / raster_layer.rasterUnitsPerPixelX())
                        pixel_y = int((ref_extent.yMaximum() - point_geom.y()) / raster_layer.rasterUnitsPerPixelY())

                        if 0 <= pixel_x < raster_layer.width() and 0 <= pixel_y < raster_layer.height():
                            points.append({
                                'x': pixel_x,
                                'y': pixel_y,
                                'coord_x': point_geom.x(),
                                'coord_y': point_geom.y(),
                                'id': point_id,
                                'ref_value': ref_val
                            })
                            reference_values_from_csv.append(ref_val)
                            point_ids.append(point_id)
                        
                    except (ValueError, IndexError) as e:
                        self.result_text.append(f"   ⚠ Satır {line_num} atlandı / Line {line_num} skipped: {str(e)}\n")
                        continue
                
                return points, reference_values_from_csv, point_ids
                
        except Exception as e:
            raise Exception(f"CSV dosyası yüklenirken hata / Error loading CSV: {str(e)}")
        
    def load_raster_layers(self, combo):
        """Raster katmanlarını yükle"""
        combo.clear()
        layers = QgsProject.instance().mapLayers().values()
        raster_layers = [layer for layer in layers if isinstance(layer, QgsRasterLayer)]
        
        for layer in raster_layers:
            combo.addItem(layer.name(), layer)

    def raster_to_array(self, layer):
        """Raster katmanının 1. bandını (height, width) numpy dizisine dönüştürür.
        NoData pikselleri NaN olarak işaretlenir. Performans için block.data()
        üzerinden vektörize okuma yapılır; desteklenmeyen/eşleşmeyen veri tipinde
        piksel piksel okumaya (yavaş ama güvenli) düşer."""
        provider = layer.dataProvider()
        extent = layer.extent()
        width = layer.width()
        height = layer.height()
        block = provider.block(1, extent, width, height)

        # Qgis.DataType sabit değerleri (QGIS 3.x/4.x arası kararlı)
        dtype_map = {
            1: np.uint8, 2: np.uint16, 3: np.int16, 4: np.uint32,
            5: np.int32, 6: np.float32, 7: np.float64, 14: np.int8,
        }
        np_dtype = dtype_map.get(int(block.dataType()))

        if np_dtype is not None:
            try:
                raw = np.frombuffer(block.data(), dtype=np_dtype)
                if raw.size == width * height:
                    array = raw.reshape((height, width)).astype(np.float64)
                    if block.hasNoDataValue():
                        array[array == block.noDataValue()] = np.nan
                    return array
            except ValueError:
                pass

        # Fallback: piksel piksel okuma
        array = np.full((height, width), np.nan)
        nodata = block.noDataValue() if block.hasNoDataValue() else None
        for y in range(height):
            for x in range(width):
                val = block.value(y, x)
                if nodata is not None and val == nodata:
                    continue
                array[y, x] = val
        return array

    def generate_sampling_points(self, reference_layer, n_points, method):
        """Örnekleme noktaları oluştur"""
        extent = reference_layer.extent()

        points = []
        max_attempts = n_points * 100
        attempts = 0
        
        if method == 'random':
            while len(points) < n_points and attempts < max_attempts:
                x = random.uniform(extent.xMinimum(), extent.xMaximum())
                y = random.uniform(extent.yMinimum(), extent.yMaximum())
                
                pixel_x = int((x - extent.xMinimum()) / reference_layer.rasterUnitsPerPixelX())
                pixel_y = int((extent.yMaximum() - y) / reference_layer.rasterUnitsPerPixelY())
                
                if 0 <= pixel_x < reference_layer.width() and 0 <= pixel_y < reference_layer.height():
                    points.append({
                        'x': pixel_x,
                        'y': pixel_y,
                        'coord_x': x,
                        'coord_y': y
                    })
                    
                attempts += 1
                
        elif method == 'systematic':
            grid_size = int(np.sqrt(n_points))
            x_step = reference_layer.width() / grid_size
            y_step = reference_layer.height() / grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(points) >= n_points:
                        break
                        
                    pixel_x = int(i * x_step + x_step / 2)
                    pixel_y = int(j * y_step + y_step / 2)
                    
                    if 0 <= pixel_x < reference_layer.width() and 0 <= pixel_y < reference_layer.height():
                        x = extent.xMinimum() + pixel_x * reference_layer.rasterUnitsPerPixelX()
                        y = extent.yMaximum() - pixel_y * reference_layer.rasterUnitsPerPixelY()
                        
                        points.append({
                            'x': pixel_x,
                            'y': pixel_y,
                            'coord_x': x,
                            'coord_y': y
                        })
                        
        elif method == 'stratified':
            reference_array = self.raster_to_array(reference_layer)
            valid_mask = ~np.isnan(reference_array) & (reference_array != -9999)

            unique_classes = np.unique(reference_array[valid_mask])
            points_per_class = n_points // len(unique_classes)
            
            for class_val in unique_classes:
                class_points = np.argwhere(reference_array == class_val)
                
                if len(class_points) > 0:
                    n_sample = min(points_per_class, len(class_points))
                    sampled_indices = np.random.choice(len(class_points), n_sample, replace=False)
                    
                    for idx in sampled_indices:
                        pixel_y, pixel_x = class_points[idx]
                        x = extent.xMinimum() + pixel_x * reference_layer.rasterUnitsPerPixelX()
                        y = extent.yMaximum() - pixel_y * reference_layer.rasterUnitsPerPixelY()
                        
                        points.append({
                            'x': int(pixel_x),
                            'y': int(pixel_y),
                            'coord_x': float(x),
                            'coord_y': float(y)
                        })
                        
        return points
        
    def run_validation(self):
        """CARAS doğrulama analizini çalıştır"""
        try:
            if self.csv_radio.isChecked():
                classified_layer = self.classified_combo.currentData()
                
                if not classified_layer:
                    QMessageBox.warning(self, "Uyarı / Warning", 
                        "Lütfen sınıflandırılmış haritayı seçin!\n"
                        "Please select the classified map!")
                    return
                    
                reference_layer = None
            else:
                reference_layer = self.reference_combo.currentData()
                classified_layer = self.classified_combo.currentData()
                
                if not reference_layer or not classified_layer:
                    QMessageBox.warning(self, "Uyarı / Warning", 
                        "Lütfen her iki haritayı da seçin!\n"
                        "Please select both maps!")
                    return
                
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.result_text.clear()
            self.result_text.append("⏳ CARAS analizi başlatılıyor...\n⏳ Starting CARAS analysis...\n")
            QApplication.processEvents()
            
            self.progress_bar.setValue(10)
            
            csv_reference_values = None
            point_ids = None
            
            if self.csv_radio.isChecked():
                csv_path = self.csv_path_edit.text()
                if not csv_path:
                    QMessageBox.warning(self, "Uyarı / Warning",
                        "Lütfen CSV dosyası seçin!\n"
                        "Please select a CSV file!")
                    self.progress_bar.setVisible(False)
                    return
                
                self.result_text.append("📍 CSV'den noktalar yükleniyor...\n📍 Loading points from CSV...\n")
                QApplication.processEvents()
                
                self.sampled_points, csv_reference_values, point_ids = self.load_points_from_csv(csv_path, classified_layer)
                
                if not self.sampled_points:
                    QMessageBox.critical(self, "Hata / Error", 
                        "CSV'den nokta yüklenemedi!\n"
                        "Could not load points from CSV!")
                    self.progress_bar.setVisible(False)
                    return
                    
                self.result_text.append(f"✓ {len(self.sampled_points)} nokta CSV'den yüklendi\n"
                                      f"✓ {len(self.sampled_points)} points loaded from CSV\n")
            else:
                self.result_text.append("📍 Örnekleme noktaları oluşturuluyor...\n📍 Generating sampling points...\n")
                QApplication.processEvents()
                
                n_points = self.points_spin.value()
                method_id = self.method_group.checkedId()
                method = {1: 'random', 2: 'stratified', 3: 'systematic'}[method_id]
                
                self.sampled_points = self.generate_sampling_points(reference_layer, n_points, method)
                
                if not self.sampled_points:
                    QMessageBox.critical(self, "Hata / Error", 
                        "Örnekleme noktaları oluşturulamadı!\n"
                        "Could not generate sampling points!")
                    self.progress_bar.setVisible(False)
                    return
                    
                self.result_text.append(f"✓ {len(self.sampled_points)} nokta oluşturuldu\n"
                                      f"✓ {len(self.sampled_points)} points generated\n")
            QApplication.processEvents()
            
            self.progress_bar.setValue(30)
            self.result_text.append("\n📊 Raster verileri okunuyor...\n📊 Reading raster data...\n")
            QApplication.processEvents()
            
            class_extent = classified_layer.extent()
            classified_data = self.raster_to_array(classified_layer)

            if not self.csv_radio.isChecked():
                ref_extent = reference_layer.extent()
                reference_data = self.raster_to_array(reference_layer)
            
            self.progress_bar.setValue(50)
            
            if csv_reference_values is not None:
                reference_values = []
                classified_values = []
                valid_points = []
                
                class_extent = classified_layer.extent()
                
                for i, point in enumerate(self.sampled_points):
                    coord_x = point['coord_x']
                    coord_y = point['coord_y']
                    
                    class_pixel_x = int((coord_x - class_extent.xMinimum()) / classified_layer.rasterUnitsPerPixelX())
                    class_pixel_y = int((class_extent.yMaximum() - coord_y) / classified_layer.rasterUnitsPerPixelY())
                    
                    if (0 <= class_pixel_x < classified_layer.width() and 
                        0 <= class_pixel_y < classified_layer.height()):
                        
                        class_val = classified_data[class_pixel_y, class_pixel_x]
                        ref_val = csv_reference_values[i]
                        
                        is_ref_valid = not (np.isnan(ref_val) or ref_val == -9999 or ref_val is None)
                        is_class_valid = not (np.isnan(class_val) or class_val == -9999 or class_val is None)
                        
                        if is_ref_valid and is_class_valid:
                            reference_values.append(ref_val)
                            classified_values.append(class_val)
                            valid_points.append(point)
                
                self.sampled_points = valid_points
                self.result_text.append(f"✓ CSV referans değerleri kullanıldı\n"
                                      f"✓ Using CSV reference values\n")
                
            else:
                reference_values = []
                classified_values = []
                
                ref_extent = reference_layer.extent()
                class_extent = classified_layer.extent()
                
                valid_points = []
                
                for point in self.sampled_points:
                    coord_x = point['coord_x']
                    coord_y = point['coord_y']
                    
                    ref_pixel_x = int((coord_x - ref_extent.xMinimum()) / reference_layer.rasterUnitsPerPixelX())
                    ref_pixel_y = int((ref_extent.yMaximum() - coord_y) / reference_layer.rasterUnitsPerPixelY())
                    
                    class_pixel_x = int((coord_x - class_extent.xMinimum()) / classified_layer.rasterUnitsPerPixelX())
                    class_pixel_y = int((class_extent.yMaximum() - coord_y) / classified_layer.rasterUnitsPerPixelY())
                    
                    if (0 <= ref_pixel_x < reference_layer.width() and 
                        0 <= ref_pixel_y < reference_layer.height() and
                        0 <= class_pixel_x < classified_layer.width() and 
                        0 <= class_pixel_y < classified_layer.height()):
                        
                        ref_val = reference_data[ref_pixel_y, ref_pixel_x]
                        class_val = classified_data[class_pixel_y, class_pixel_x]
                        
                        is_ref_valid = not (np.isnan(ref_val) or ref_val == -9999 or ref_val is None)
                        is_class_valid = not (np.isnan(class_val) or class_val == -9999 or class_val is None)
                        
                        if is_ref_valid and is_class_valid:
                            reference_values.append(ref_val)
                            classified_values.append(class_val)
                            valid_points.append(point)
                
                self.sampled_points = valid_points
            
            if len(reference_values) == 0:
                QMessageBox.critical(self, "Hata / Error", 
                    "Geçerli örnekleme noktası bulunamadı!\n"
                    "No valid sampling points found!\n"
                    "Raster haritalarının extent ve CRS değerlerini kontrol edin.")
                self.progress_bar.setVisible(False)
                return
            
            self.result_text.append(f"✓ {len(reference_values)} geçerli nokta kullanılıyor\n"
                                  f"✓ Using {len(reference_values)} valid points\n")
            QApplication.processEvents()
            
            self.result_text.append("\n🔍 Tüm sınıf değerleri okunuyor...\n🔍 Reading all class values...\n")
            QApplication.processEvents()
            
            class_valid_mask = ~np.isnan(classified_data) & (classified_data != -9999)
            class_unique_values = set(classified_data[class_valid_mask].tolist())

            if csv_reference_values is not None:
                ref_unique_values = set(reference_values)
            else:
                ref_valid_mask = ~np.isnan(reference_data) & (reference_data != -9999)
                ref_unique_values = set(reference_data[ref_valid_mask].tolist())
            
            self.result_text.append(f"✓ Referans: {len(ref_unique_values)} benzersiz sınıf\n")
            self.result_text.append(f"✓ Reference: {len(ref_unique_values)} unique classes\n")
            self.result_text.append(f"✓ Sınıflandırılmış: {len(class_unique_values)} benzersiz sınıf\n")
            self.result_text.append(f"✓ Classified: {len(class_unique_values)} unique classes\n")
            QApplication.processEvents()
            
            self.result_text.append("\n🔄 Sınıf eşleştirme bekleniyor...\n🔄 Waiting for class mapping...\n")
            QApplication.processEvents()
            
            mapping_dialog = ClassMappingDialog(list(ref_unique_values), list(class_unique_values), self)
            if mapping_dialog.exec() != QDialog.DialogCode.Accepted:
                self.progress_bar.setVisible(False)
                self.result_text.append("\n❌ Analiz iptal edildi\n❌ Analysis cancelled\n")
                return
                
            reference_mapping, classified_mapping, class_names = mapping_dialog.get_mappings()
            
            self.progress_bar.setValue(60)
            self.result_text.append("\n🔢 Sınıf kategorileri uygulanıyor...\n🔢 Applying class categories...\n")
            QApplication.processEvents()
            
            all_categories = sorted(set(list(reference_mapping.values()) + list(classified_mapping.values())))
            
            reference_categories = []
            classified_categories = []
            
            for ref_val, class_val in zip(reference_values, classified_values):
                if ref_val in reference_mapping and class_val in classified_mapping:
                    reference_categories.append(reference_mapping[ref_val])
                    classified_categories.append(classified_mapping[class_val])
            
            sorted_categories = sorted(all_categories)
            category_labels = [class_names.get(cat, f"Kategori_{cat}") for cat in sorted_categories]
            
            self.result_text.append(f"✓ Toplam {len(all_categories)} kategori tanımlandı\n")
            self.result_text.append(f"✓ Total {len(all_categories)} categories defined\n")
            for cat in sorted_categories:
                self.result_text.append(f"  - Kategori {cat}: {class_names.get(cat, f'Kategori_{cat}')}\n")
            QApplication.processEvents()
            
            self.progress_bar.setValue(80)
            self.result_text.append("\n📈 Metrikler hesaplanıyor...\n📈 Calculating metrics...\n")
            QApplication.processEvents()
            
            cm = confusion_matrix(reference_categories, classified_categories, labels=sorted_categories)
            overall_accuracy = accuracy_score(reference_categories, classified_categories)
            kappa = cohen_kappa_score(reference_categories, classified_categories)
            
            f1_macro = f1_score(reference_categories, classified_categories, 
                               labels=sorted_categories, average='macro', zero_division=0)
            f1_weighted = f1_score(reference_categories, classified_categories, 
                                  labels=sorted_categories, average='weighted', zero_division=0)
            precision_macro = precision_score(reference_categories, classified_categories, 
                                             labels=sorted_categories, average='macro', zero_division=0)
            recall_macro = recall_score(reference_categories, classified_categories, 
                                       labels=sorted_categories, average='macro', zero_division=0)
            
            ref_arr = np.array(reference_values, dtype=float)
            cls_arr = np.array(classified_values, dtype=float)
            
            r2 = r2_score(ref_arr, cls_arr)
            rmse = np.sqrt(mean_squared_error(ref_arr, cls_arr))
            mae = mean_absolute_error(ref_arr, cls_arr)
            bias = float(np.mean(cls_arr - ref_arr))
            
            ref_cat_arr = np.array(reference_categories, dtype=float)
            cls_cat_arr = np.array(classified_categories, dtype=float)
            
            r2_cat = r2_score(ref_cat_arr, cls_cat_arr)
            rmse_cat = np.sqrt(mean_squared_error(ref_cat_arr, cls_cat_arr))
            mae_cat = mean_absolute_error(ref_cat_arr, cls_cat_arr)
            bias_cat = float(np.mean(cls_cat_arr - ref_cat_arr))
            
            class_report = classification_report(
                reference_categories, 
                classified_categories,
                labels=sorted_categories,
                target_names=category_labels,
                zero_division=0,
                output_dict=True
            )
            
            self.progress_bar.setValue(90)
            
            self.validation_results = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'plugin': 'CARAS - Classification Accuracy and Regression Assessment Suite',
                'reference_map': 'CSV Data' if csv_reference_values is not None else reference_layer.name(),
                'classified_map': classified_layer.name(),
                'n_points': len(reference_categories),
                'sampling_method': 'CSV File' if csv_reference_values is not None else method,
                'overall_accuracy': float(overall_accuracy),
                'kappa': float(kappa),
                'f1_macro': float(f1_macro),
                'f1_weighted': float(f1_weighted),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'bias': float(bias),
                'r2_cat': float(r2_cat),
                'rmse_cat': float(rmse_cat),
                'mae_cat': float(mae_cat),
                'bias_cat': float(bias_cat),
                'confusion_matrix': cm.tolist(),
                'class_names': category_labels,
                'class_report': class_report,
                'all_categories': sorted_categories
            }
            
            self.display_results()
            
            self.progress_bar.setValue(100)
            self.export_button.setEnabled(True)
            self.save_points_button.setEnabled(True)
            
            QMessageBox.information(self, "Başarılı / Success", 
                "✓ CARAS analizi tamamlandı!\n"
                "✓ CARAS analysis completed!")
            
        except Exception as e:
            QMessageBox.critical(self, "Hata / Error", 
                f"Analiz sırasında hata oluştu / Error during analysis:\n{str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            
    def display_results(self):
        """Sonuçları göster"""
        results = self.validation_results
        
        output = "=" * 80 + "\n"
        output += "CARAS — Classification Accuracy and Regression Assessment Suite\n"
        output += "=" * 80 + "\n\n"
        
        output += f"📅 Analiz Tarihi / Analysis Date: {results['timestamp']}\n"
        output += f"📍 Nokta Sayısı / Number of Points: {results['n_points']}\n"
        output += f"🎯 Örnekleme Metodu / Sampling Method: {results['sampling_method'].upper()}\n\n"
        
        output += "-" * 80 + "\n"
        output += "PRIMARY METRICS / TEMEL METRİKLER\n"
        output += "-" * 80 + "\n"
        output += f"Overall Accuracy (OA)       : {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)\n"
        output += f"Cohen's Kappa (κ)           : {results['kappa']:.4f}\n"
        output += f"F1-Score (Macro)            : {results['f1_macro']:.4f}\n"
        output += f"F1-Score (Weighted)         : {results['f1_weighted']:.4f}\n"
        output += f"Precision (Macro)           : {results['precision_macro']:.4f}\n"
        output += f"Recall (Macro)              : {results['recall_macro']:.4f}\n\n"
        
        kappa_val = results['kappa']
        if kappa_val < 0:
            kappa_interp = "Poor (Zayıf)"
        elif kappa_val < 0.20:
            kappa_interp = "Slight (Hafif)"
        elif kappa_val < 0.40:
            kappa_interp = "Fair (Orta)"
        elif kappa_val < 0.60:
            kappa_interp = "Moderate (İyi)"
        elif kappa_val < 0.80:
            kappa_interp = "Substantial (Çok İyi)"
        else:
            kappa_interp = "Almost Perfect (Mükemmel)"
            
        output += f"Kappa Interpretation        : {kappa_interp}\n\n"
        
        output += "-" * 80 + "\n"
        output += "REGRESSION STATISTICS (Raw Pixel Values) / REGRESYON İSTATİSTİKLERİ (Ham Piksel)\n"
        output += "-" * 80 + "\n"
        output += f"R² (Coeff. of Determination): {results['r2']:.4f}\n"
        output += f"RMSE (Root Mean Sq. Error) : {results['rmse']:.4f}\n"
        output += f"MAE  (Mean Absolute Error)  : {results['mae']:.4f}\n"
        output += f"Bias (Mean Error)           : {results['bias']:.4f}"
        bias_dir = " (Overestimation / Fazla Tahmin)" if results['bias'] > 0 else " (Underestimation / Az Tahmin)" if results['bias'] < 0 else " (No Bias / Sapma Yok)"
        output += f"{bias_dir}\n\n"
        
        output += "-" * 80 + "\n"
        output += "REGRESSION STATISTICS (Category Values) / REGRESYON İSTATİSTİKLERİ (Kategori)\n"
        output += "-" * 80 + "\n"
        output += f"R² (Coeff. of Determination): {results['r2_cat']:.4f}\n"
        output += f"RMSE (Root Mean Sq. Error) : {results['rmse_cat']:.4f}\n"
        output += f"MAE  (Mean Absolute Error)  : {results['mae_cat']:.4f}\n"
        output += f"Bias (Mean Error)           : {results['bias_cat']:.4f}"
        bias_dir_cat = " (Overestimation / Fazla Tahmin)" if results['bias_cat'] > 0 else " (Underestimation / Az Tahmin)" if results['bias_cat'] < 0 else " (No Bias / Sapma Yok)"
        output += f"{bias_dir_cat}\n\n"
        output += "-" * 80 + "\n"
        
        cm = np.array(results['confusion_matrix'])
        class_names = results['class_names']
        
        header = "Reference \\ Predicted".ljust(25)
        for name in class_names:
            header += f"{name[:12]:>14}"
        output += header + "\n"
        output += "-" * 80 + "\n"
        
        for i, row in enumerate(cm):
            line = f"{class_names[i][:23]:23}  "
            for val in row:
                line += f"{val:>14}"
            output += line + "\n"
            
        output += "\n"
        
        output += "-" * 80 + "\n"
        output += "PER-CLASS METRICS / SINIF BAZLI METRİKLER\n"
        output += "-" * 80 + "\n"
        
        class_report = results['class_report']
        
        output += f"{'Class/Sınıf':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n"
        output += "-" * 80 + "\n"
        
        for class_name in class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                output += f"{class_name:<25} "
                output += f"{metrics['precision']:<12.4f} "
                output += f"{metrics['recall']:<12.4f} "
                output += f"{metrics['f1-score']:<12.4f} "
                output += f"{int(metrics['support']):<10}\n"
                
        output += "\n"
        
        output += "-" * 80 + "\n"
        output += "PRODUCER'S & USER'S ACCURACY / ÜRETİCİ VE KULLANICI DOĞRULUĞU\n"
        output += "-" * 80 + "\n"
        
        output += f"{'Class/Sınıf':<25} {'Producer Acc.':<15} {'User Acc.':<15}\n"
        output += "-" * 80 + "\n"
        
        for i, class_name in enumerate(class_names):
            if class_name in class_report:
                producer_acc = class_report[class_name]['recall']
                user_acc = class_report[class_name]['precision']
                
                output += f"{class_name:<25} "
                output += f"{producer_acc:<15.4f} "
                output += f"{user_acc:<15.4f}\n"
                
        output += "\n"
        output += "=" * 80 + "\n"
        output += "Generated by CARAS — Classification Accuracy and Regression Assessment Suite\n"
        output += "=" * 80 + "\n"
        
        self.result_text.setPlainText(output)
        
    def save_validation_points(self):
        """Doğrulama noktalarını shapefile olarak kaydet"""
        if not self.sampled_points or not self.validation_results:
            QMessageBox.warning(self, "Uyarı / Warning", 
                "Önce CARAS analizi yapmalısınız!\n"
                "You must run CARAS analysis first!")
            return
            
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Noktaları Kaydet / Save Points", 
                f"caras_validation_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.shp", 
                "Shapefile (*.shp)"
            )
            
            if not file_path:
                return
                
            if _USE_QVARIANT:
                fields = [
                    QgsField("point_id", QVariant.Int),
                    QgsField("ref_value", QVariant.Double),
                    QgsField("class_value", QVariant.Double),
                    QgsField("match", QVariant.String)
                ]
            else:
                from qgis.core import QgsFields
                fields = [
                    QgsField("point_id", type=2),
                    QgsField("ref_value", type=6),
                    QgsField("class_value", type=6),
                    QgsField("match", type=10)
                ]
            
            classified_layer = self.classified_combo.currentData()
            crs = classified_layer.crs()
            
            vector_layer = QgsVectorLayer(f"Point?crs={crs.authid()}", "caras_validation_points", "memory")
            provider = vector_layer.dataProvider()
            provider.addAttributes(fields)
            vector_layer.updateFields()
            
            features = []
            
            is_csv = self.validation_results['reference_map'] == 'CSV Data'
            
            if is_csv:
                class_extent = classified_layer.extent()
                classified_data = self.raster_to_array(classified_layer)

                for i, point in enumerate(self.sampled_points):
                    feature = QgsFeature()
                    feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(point['coord_x'], point['coord_y'])))
                    
                    coord_x = point['coord_x']
                    coord_y = point['coord_y']
                    
                    class_pixel_x = int((coord_x - class_extent.xMinimum()) / classified_layer.rasterUnitsPerPixelX())
                    class_pixel_y = int((class_extent.yMaximum() - coord_y) / classified_layer.rasterUnitsPerPixelY())
                    
                    class_val = classified_data[class_pixel_y, class_pixel_x]
                    ref_val = point.get('ref_value', i+1)
                    
                    match = "Yes" if abs(ref_val - class_val) < 0.001 else "No"
                    
                    feature.setAttributes([i+1, float(ref_val), float(class_val), match])
                    features.append(feature)
            else:
                reference_layer = self.reference_combo.currentData()
                
                ref_extent = reference_layer.extent()
                class_extent = classified_layer.extent()

                reference_data = self.raster_to_array(reference_layer)
                classified_data = self.raster_to_array(classified_layer)

                for i, point in enumerate(self.sampled_points):
                    feature = QgsFeature()
                    feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(point['coord_x'], point['coord_y'])))
                    
                    coord_x = point['coord_x']
                    coord_y = point['coord_y']
                    
                    ref_pixel_x = int((coord_x - ref_extent.xMinimum()) / reference_layer.rasterUnitsPerPixelX())
                    ref_pixel_y = int((ref_extent.yMaximum() - coord_y) / reference_layer.rasterUnitsPerPixelY())
                    
                    class_pixel_x = int((coord_x - class_extent.xMinimum()) / classified_layer.rasterUnitsPerPixelX())
                    class_pixel_y = int((class_extent.yMaximum() - coord_y) / classified_layer.rasterUnitsPerPixelY())
                    
                    ref_val = reference_data[ref_pixel_y, ref_pixel_x]
                    class_val = classified_data[class_pixel_y, class_pixel_x]
                    match = "Yes" if abs(ref_val - class_val) < 0.001 else "No"
                    
                    feature.setAttributes([i+1, float(ref_val), float(class_val), match])
                    features.append(feature)
                
            provider.addFeatures(features)
            
            save_options = QgsVectorFileWriter.SaveVectorOptions()
            save_options.driverName = "ESRI Shapefile"
            save_options.fileEncoding = "UTF-8"
            
            error = QgsVectorFileWriter.writeAsVectorFormatV3(
                vector_layer,
                file_path,
                QgsCoordinateTransformContext(),
                save_options
            )
            
            if error[0] == QgsVectorFileWriter.WriterError.NoError:
                saved_layer = QgsVectorLayer(file_path, "CARAS Validation Points", "ogr")
                QgsProject.instance().addMapLayer(saved_layer)
                
                QMessageBox.information(self, "Başarılı / Success", 
                    f"✓ Noktalar başarıyla kaydedildi!\n"
                    f"✓ Points saved successfully!\n\n{file_path}")
            else:
                QMessageBox.critical(self, "Hata / Error", 
                    f"Noktalar kaydedilirken hata oluştu / Error saving points:\n{error[1]}")
                    
        except Exception as e:
            QMessageBox.critical(self, "Hata / Error", 
                f"Noktalar kaydedilirken hata oluştu / Error saving points:\n{str(e)}")
            
    def export_results(self):
        """Sonuçları dosyaya aktar"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Raporu Kaydet / Save Report", 
                f"caras_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                "Text File (*.txt);;JSON File (*.json);;HTML Report (*.html)"
            )
            
            if not file_path:
                return
                
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
                    
            elif file_path.endswith('.html'):
                html_content = self.generate_html_report()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                    
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.result_text.toPlainText())
                    
            QMessageBox.information(self, "Başarılı / Success", 
                f"Rapor başarıyla kaydedildi!\n"
                f"Report saved successfully!\n\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Hata / Error", 
                f"Rapor kaydedilirken hata oluştu / Error saving report:\n{str(e)}")
                
    def generate_html_report(self):
        """HTML raporu oluştur"""
        results = self.validation_results
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CARAS — Classification Accuracy and Regression Assessment Suite</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 40px; 
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 4px solid #3498db; 
            padding-bottom: 10px;
        }}
        .subtitle {{
            color: #7f8c8d;
            font-size: 0.95em;
            margin-top: -10px;
            margin-bottom: 20px;
        }}
        h2 {{ 
            color: #34495e; 
            margin-top: 30px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }}
        .metric {{ 
            background: #ecf0f1; 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 8px;
            border-left: 5px solid #3498db;
        }}
        .metric-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: center; 
        }}
        th {{ 
            background-color: #3498db; 
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌍 CARAS Report</h1>
        <p class="subtitle">Classification Accuracy and Regression Assessment Suite</p>
        <p><strong>Analysis Date:</strong> {results['timestamp']}</p>
        
        <h2>📊 Primary Metrics / Temel Metrikler</h2>
        <div class="metric">
            <p><strong>Overall Accuracy (OA):</strong> 
            <span class="metric-value">{results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)</span></p>
        </div>
        <div class="metric">
            <p><strong>Cohen's Kappa (κ):</strong> 
            <span class="metric-value">{results['kappa']:.4f}</span></p>
        </div>
        <div class="metric">
            <p><strong>F1-Score (Macro):</strong> 
            <span class="metric-value">{results['f1_macro']:.4f}</span></p>
            <p><strong>F1-Score (Weighted):</strong> 
            <span class="metric-value">{results['f1_weighted']:.4f}</span></p>
        </div>
        <div class="metric">
            <p><strong>Precision (Macro):</strong> {results['precision_macro']:.4f}</p>
            <p><strong>Recall (Macro):</strong> {results['recall_macro']:.4f}</p>
        </div>
        
        <h2>📐 Regression Statistics / Regresyon İstatistikleri</h2>
        <div class="metric">
            <p><em>Ham Piksel Değerleri / Raw Pixel Values</em></p>
            <p><strong>R² (Determination Coeff.):</strong> <span class="metric-value">{results['r2']:.4f}</span></p>
            <p><strong>RMSE (Root Mean Sq. Error):</strong> <span class="metric-value">{results['rmse']:.4f}</span></p>
            <p><strong>MAE (Mean Absolute Error):</strong> <span class="metric-value">{results['mae']:.4f}</span></p>
            <p><strong>Bias (Mean Error):</strong> <span class="metric-value">{results['bias']:.4f}</span>
            {"&nbsp;⬆ Overestimation" if results['bias'] > 0 else "&nbsp;⬇ Underestimation" if results['bias'] < 0 else "&nbsp;✓ No Bias"}</p>
        </div>
        <div class="metric">
            <p><em>Kategori Değerleri / Category Values</em></p>
            <p><strong>R²:</strong> <span class="metric-value">{results['r2_cat']:.4f}</span></p>
            <p><strong>RMSE:</strong> <span class="metric-value">{results['rmse_cat']:.4f}</span></p>
            <p><strong>MAE:</strong> <span class="metric-value">{results['mae_cat']:.4f}</span></p>
            <p><strong>Bias:</strong> <span class="metric-value">{results['bias_cat']:.4f}</span>
            {"&nbsp;⬆ Overestimation" if results['bias_cat'] > 0 else "&nbsp;⬇ Underestimation" if results['bias_cat'] < 0 else "&nbsp;✓ No Bias"}</p>
        </div>
        
        <h2>📋 Confusion Matrix / Karmaşıklık Matrisi</h2>
        <table>
            <tr>
                <th>Reference \\ Predicted</th>
"""
        
        class_names = results['class_names']
        cm = np.array(results['confusion_matrix'])
        
        for name in class_names:
            html += f"<th>{name}</th>"
        html += "</tr>\n"
        
        for i, row in enumerate(cm):
            html += f"<tr><th>{class_names[i]}</th>"
            for val in row:
                html += f"<td>{val}</td>"
            html += "</tr>\n"
                
        html += """
        </table>
        
        <h2>💡 Quality Assessment / Kalite Değerlendirmesi</h2>
        <p>Detailed analysis results are available in the complete report.</p>
        
        <div class="footer">
            <p>Generated by <strong>CARAS — Classification Accuracy and Regression Assessment Suite</strong></p>
            <p>QGIS Plugin | Author: Ömer K. ÖRÜCÜ</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html


class CARASPlugin:
    """CARAS QGIS Plugin Sınıfı"""
    def __init__(self, iface):
        self.iface = iface
        self.dialog = None
        self.action = None
        
    def initGui(self):
        """Plugin GUI'sini başlat"""
        self.action = QAction("CARAS — Accuracy & Regression Assessment", self.iface.mainWindow())
        self.action.setToolTip("CARAS — Classification Accuracy and Regression Assessment Suite")
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu("&CARAS", self.action)
        self.iface.addToolBarIcon(self.action)
        
    def unload(self):
        """Plugin'i kaldır"""
        self.iface.removePluginMenu("&CARAS", self.action)
        self.iface.removeToolBarIcon(self.action)
        
    def run(self):
        """Plugin'i çalıştır"""
        if self.dialog is None:
            self.dialog = CARASDialog()
        
        self.dialog.load_raster_layers(self.dialog.reference_combo)
        self.dialog.load_raster_layers(self.dialog.classified_combo)
        
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()


def classFactory(iface):
    """QGIS plugin factory"""
    return CARASPlugin(iface)
