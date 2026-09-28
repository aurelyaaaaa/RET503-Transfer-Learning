# Transfer Learning - RET503 Pertemuan 3

## Proyek
Klasifikasi komponen mekanik untuk kebutuhan persepsi robot. Dataset yang digunakan terdiri dari 4 kelas:
- bolt
- nut
- locatingpin
- washer

Dataset ini dipilih karena klasifikasi komponen mekanik dapat digunakan sebagai dasar sistem visi robot untuk mengenali komponen yang akan dipindahkan, disortir, atau diperiksa.

## Sumber data
Dataset publik **Images of mechanical parts (Bolt, Nut, Washer, Pin)** dari Kaggle:
https://www.kaggle.com/datasets/manikantanrnair/images-of-mechanical-parts-boltnut-washerpin

Dataset tersebut berisi 4 kelas dan masing-masing kelas memiliki banyak gambar, sehingga memenuhi kebutuhan minimal 50 citra per kelas. Untuk tugas ini, script persiapan mengambil 50 citra per kelas agar eksperimen lebih ringan.

Catatan: sebelum dikumpulkan, cek kembali lisensi dan ketentuan penggunaan dataset pada halaman sumber.

## Metode
Eksperimen menggunakan **ResNet-18 pretrained ImageNet** dan membandingkan:
1. **Feature extraction**: backbone dibekukan, hanya `fc` yang dilatih, learning rate `1e-3`.
2. **Fine-tuning parsial**: `layer4` + `fc` dilatih, learning rate `1e-4` untuk `layer4` dan `1e-3` untuk `fc`.
3. **Training from scratch**: seluruh model dilatih dari bobot acak, learning rate `1e-3`.

Konfigurasi mengikuti materi RET503 Pertemuan 3: augmentasi `RandomResizedCrop`, `HorizontalFlip`, `ColorJitter`, 10 epoch, dan `CosineAnnealingLR`.

## Struktur folder
```text
transfer-learning-ret503/
├── dataset_raw/
│   ├── bolt/
│   ├── nut/
│   ├── locatingpin/
│   └── washer/
├── results/
├── download_and_prepare.py
├── split.py
├── train.py
├── latency.py
├── requirements.txt
├── metadata.csv
└── README.md
```

## Cara menjalankan
### 1. Install library
```bash
pip install -r requirements.txt
```

### 2. Siapkan dataset
Cara paling mudah di Google Colab:
```bash
python download_and_prepare.py
```

Script akan mencoba mengunduh dataset melalui `kagglehub`, kemudian mengambil 50 citra per kelas dan membuat `metadata.csv`.

Jika `kagglehub` meminta autentikasi, dataset juga dapat diunduh manual dari halaman Kaggle lalu folder 4 kelas diletakkan di `dataset_raw/`.

### 3. Bagi train/validation/test
```bash
python split.py
```

Pembagian default:
- 70% train
- 15% validation
- 15% test

Pemisahan dilakukan secara acak dengan seed tetap agar hasil dapat direproduksi.

### 4. Latih 3 pendekatan
```bash
python train.py --mode feature
python train.py --mode partial
python train.py --mode scratch
```

Hasil akan disimpan di folder `results/`.

### 5. Ukur latensi
```bash
python latency.py
```

Script membandingkan latensi inferensi ResNet-18 dengan MobileNetV3-Small menggunakan gambar uji.

## Output yang perlu dikumpulkan
- dataset_raw + `metadata.csv`
- dokumen desain awal
- hasil 3 mode
- grafik akurasi per epoch
- latensi model yang dipilih
- README berisi analisis singkat

## Catatan penting
Jangan memasukkan frame yang hampir sama ke train dan validation. Materi kuliah memperingatkan bahwa frame berurutan yang sangat mirip dapat menyebabkan **data leakage** dan membuat akurasi validasi terlihat terlalu bagus.
