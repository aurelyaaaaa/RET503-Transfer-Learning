# Dokumen Desain Awal

## 1. Misi proyek
Model visi digunakan untuk mengenali jenis komponen mekanik dari citra kamera. Hasil klasifikasi dapat menjadi bagian dari pipeline persepsi robot untuk proses identifikasi, penyortiran, atau pemeriksaan komponen.

## 2. Kelas objek
- bolt
- nut
- locatingpin
- washer

Target awal: minimal 50 citra per kelas.

## 3. Kamera dan data
Dataset menggunakan citra komponen mekanik berukuran 224×224. Untuk deployment pada robot, preprocessing harus dibuat konsisten dengan preprocessing saat training.

## 4. Unit komputasi
Eksperimen dapat dijalankan di Google Colab atau komputer dengan GPU. Untuk perangkat robot/edge, model ringan seperti MobileNetV3-Small perlu dipertimbangkan karena kebutuhan komputasinya lebih rendah.

## 5. Target kinerja
- validation accuracy
- test accuracy
- waktu training
- epoch ketika validation accuracy mencapai ≥90% jika tercapai
- latensi inferensi
- FPS

## 6. Kandidat model
**ResNet-18** dipilih sebagai model utama karena materi praktikum menggunakan ResNet-18 dan model ini relatif ringan untuk eksperimen. MobileNetV3-Small digunakan sebagai pembanding latensi.

## 7. Strategi transfer learning
Eksperimen:
- feature extraction: backbone beku, hanya FC dilatih
- fine-tuning parsial: layer4 dan FC dilatih
- scratch: seluruh model dilatih dari bobot acak sebagai pembanding

## 8. Rencana data
Gunakan 50 citra per kelas untuk eksperimen awal. Data dibagi menjadi train, validation, dan test. Data train diberi augmentasi RandomResizedCrop, HorizontalFlip, dan ColorJitter.

## 9. Risiko dan mitigasi
1. **Data leakage**: data yang sangat mirip dapat masuk ke train dan validation. Mitigasi: gunakan pembagian data yang konsisten dan jangan mencampur gambar yang hampir identik.
2. **Data terlalu sedikit**: model dapat overfitting. Mitigasi: augmentasi pada data train.
3. **Model terlalu berat untuk robot**: latensi terlalu tinggi. Mitigasi: bandingkan dengan MobileNetV3-Small.
4. **Perbedaan domain**: gambar dataset berbeda dengan kamera robot. Mitigasi: tambahkan data dari lingkungan operasi robot jika tersedia.
