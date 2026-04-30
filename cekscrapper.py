from google_play_scraper import reviews, Sort
from transformers import pipeline
import csv

# =========================================================
# 1. SCRAPING DATA (Ambil ulasan dari Google Play)
# =========================================================
print("1. Sedang mengambil data ulasan dari JKN Mobile...")

result, _ = reviews(
    'app.bpjs.mobile',
    lang='id',
    country='id',
    sort=Sort.NEWEST,
    count=100,  # Anda bisa ubah jumlahnya sesuai kebutuhan UTS
    filter_score_with=None
)

# =========================================================
# 2. SENTIMENT ANALYSIS (Proses dengan Hugging Face)
# =========================================================
print("2. Sedang memuat model NLP IndoBERT (Bahasa Indonesia)...")
# Ini akan mendownload model otomatis saat pertama kali dijalankan
model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
sentiment_pipe = pipeline("sentiment-analysis", model=model_name)

print("3. Sedang menganalisis sentimen setiap ulasan...")
hasil_final = []

for review in result:
    teks_ulasan = review['content']
    
    # Batasi teks agar tidak error (max 512 karakter)
    clean_text = teks_ulasan[:512] if len(teks_ulasan) > 512 else teks_ulasan
    
    # Prediksi Sentimen
    prediksi = sentiment_pipe(clean_text)[0]
    
    # Gabungkan data asli dengan hasil analisis
    review_baru = {
        'userName': review['userName'],
        'score': review['score'],
        'at': review['at'],
        'content': teks_ulasan,
        'sentiment': prediksi['label'],      # Hasil: positive/neutral/negative
        'confidence': round(prediksi['score'], 4) # Tingkat keyakinan AI
    }
    hasil_final.append(review_baru)

# =========================================================
# 3. SIMPAN KE CSV
# =========================================================
filename = 'hasil_analisis_sentimen_jkn.csv'
header = ['userName', 'score', 'at', 'content', 'sentiment', 'confidence']

with open(filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(hasil_final)

print("-" * 30)
print(f"SELESAI! Berhasil menganalisis {len(hasil_final)} ulasan.")
print(f"File disimpan dengan nama: {filename}")