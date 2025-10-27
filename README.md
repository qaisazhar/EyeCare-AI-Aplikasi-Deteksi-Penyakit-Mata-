# Deteksi Penyakit Mata (Eye Disease Detection)

Capstone Project untuk mendeteksi penyakit mata menggunakan kecerdasan buatan. Aplikasi web ini dapat mengklasifikasikan gambar mata ke dalam empat kategori: katarak, retinopati diabetik, glaukoma, atau normal.

## Fitur

- **Upload Gambar**: Unggah gambar mata dalam format JPG, JPEG, atau PNG.
- **Prediksi Real-time**: Menggunakan model TensorFlow Lite untuk klasifikasi cepat.
- **Tampilan Hasil**: Menampilkan hasil prediksi dengan tingkat kepercayaan dan probabilitas detail.
- **Edukasi Penyakit**: Informasi lengkap tentang gejala, penyebab, dan pencegahan untuk setiap penyakit mata.
- **Interface Interaktif**: Navigasi tab untuk mempelajari tentang berbagai penyakit mata.

## Teknologi yang Digunakan

- **Streamlit**: Framework untuk membuat aplikasi web interaktif.
- **TensorFlow Lite**: Model machine learning yang dioptimalkan untuk inferensi cepat.
- **PIL (Pillow)**: Untuk pemrosesan gambar.
- **NumPy**: Untuk manipulasi array dan probabilitas.
- **EfficientNetB3**: Arsitektur model deep learning yang digunakan untuk pelatihan.

## Instalasi

1. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Pastikan Model Tersedia**:
   Pastikan file `model/model_katarak.tflite` ada di direktori `model/`.

## Penggunaan

1. **Jalankan Aplikasi**:
   ```bash
   streamlit run app.py
   ```

2. **Akses Aplikasi**:
   Buka browser dan navigasi ke `http://localhost:8501`.

3. **Upload Gambar**:
   - Klik tombol "Upload gambar mata".
   - Pilih file gambar mata (JPG, JPEG, atau PNG).
   - Tunggu hasil prediksi muncul.

4. **Jelajahi Edukasi**:
   - Gunakan tab navigasi untuk mempelajari tentang katarak, retinopati diabetik, glaukoma, atau mata normal.

## Detail Model

- **Arsitektur**: EfficientNetB3 dengan lapisan tambahan untuk klasifikasi 4 kelas.
- **Input Size**: 224x224 piksel, RGB.
- **Kelas Output**:
  - Cataract (Katarak)
  - Diabetic Retinopathy (Retinopati Diabetik)
  - Glaucoma (Glaukoma)
  - Normal
- **Akurasi**: Dilatih pada dataset gambar mata dengan performa yang baik (lihat notebook untuk detail).

## Struktur Proyek

```
project-capstone/
├── app.py                    # Aplikasi Streamlit utama
├── Capstone_Project_(deteksi_mata_katarak).ipynb  # Notebook pelatihan model
├── convert_to_tflite.py      # Script konversi model ke TFLite
├── requirements.txt          # Dependensi Python
├── README.md                 # Dokumentasi proyek (file ini)
└── model/
    ├── model_katarak.keras   # Model Keras asli
    └── model_katarak.tflite  # Model TFLite untuk inferensi
```

## Pelatihan Model

Untuk melatih ulang model atau memahami proses pelatihan:

1. Buka `Capstone_Project_(deteksi_mata_katarak).ipynb` di Jupyter Notebook atau Google Colab.
2. Pastikan dataset tersedia (format yang sama seperti di notebook).
3. Jalankan sel-sel secara berurutan untuk preprocessing, pelatihan, dan evaluasi.
4. Gunakan `convert_to_tflite.py` untuk mengkonversi model terlatih ke format TFLite.

## Kontribusi

Kontribusi untuk proyek ini sangat diterima! Silakan buat issue atau pull request untuk perbaikan atau fitur baru.

## Lisensi

Proyek ini menggunakan lisensi MIT. Lihat file LICENSE untuk detail lebih lanjut.

## Penulis

- **Nama**: Qaish shavarya azhar
- **Tahun**: 2025

---

**Catatan**: Aplikasi ini hanya untuk tujuan edukasi dan tidak menggantikan diagnosis medis profesional. Selalu konsultasikan dengan dokter mata untuk diagnosis yang akurat.
