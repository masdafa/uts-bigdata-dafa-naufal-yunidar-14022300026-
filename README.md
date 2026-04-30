# 📊 Analisis Sentimen Ulasan JKN Mobile (Google Play Store)

Proyek ini dibuat untuk memenuhi tugas **UTS Mata Kuliah Big Data / Sistem Informasi**. Sistem ini melakukan penarikan data (scraping) ulasan aplikasi **JKN Mobile** secara real-time dan mengklasifikasikannya ke dalam sentimen positif, netral, atau negatif menggunakan kecerdasan buatan. 
Nama : Dafa Naufal Yunidar (14022300026)

---

## 🌟 Fitur Utama
- **Real-time Scraping**: Mengambil data ulasan terbaru langsung dari Google Play Store.
- **Deep Learning Analysis**: Menggunakan model Transformer `IndoBERT` (Indonesian RoBERTa) untuk akurasi analisis bahasa Indonesia yang tinggi.
- **Auto-Export**: Hasil analisis otomatis disimpan ke dalam file CSV yang siap diolah lebih lanjut.

## 🛠️ Teknologi & Library
- **Bahasa**: Python 3.12
- **Library Utama**:
  - `google-play-scraper`: Mengambil data ulasan tanpa perlu API resmi.
  - `transformers`: Framework dari Hugging Face untuk menjalankan model NLP.
  - `torch`: Engine untuk pemrosesan tensor model AI.
  - `csv` & `os`: Manajemen file dan penyimpanan data.

## 🚀 Cara Menjalankan

### 1. Instalasi Dependensi
Pastikan Anda berada di direktori proyek, lalu jalankan:
```bash
pip install google-play-scraper transformers torch