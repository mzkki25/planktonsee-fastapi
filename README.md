# PlanktonSee

PlanktonSee adalah aplikasi berbasis web yang menggunakan model machine learning untuk melakukan prediksi dan klasifikasi plankton. Aplikasi ini dibangun menggunakan Python dan framework Flask serta mendukung penyimpanan data menggunakan Google Cloud.

## Struktur Direktori

```
.
├── __pycache__/         # Cache Python
├── metrics/            # Folder untuk menyimpan metrik model
├── model/              # Folder untuk menyimpan model ML
├── static/             # Folder untuk aset statis (CSS, JS, gambar)
├── templates/          # Folder untuk template HTML (Jinja2)
├── ultralytics/        # Folder untuk library YOLO (jika digunakan)
├── .gitignore          # Daftar file/folder yang diabaikan oleh Git
├── Dockerfile          # Konfigurasi untuk containerisasi aplikasi
├── README.md           # Dokumentasi proyek
├── main.py             # Entry point aplikasi Flask
├── plankton_predict.ipynb  # Notebook untuk eksplorasi model
├── plankton_predict.py # Script untuk prediksi plankton
├── railway.json        # Konfigurasi untuk deployment di Railway
├── requirements.txt    # Daftar dependensi Python
└── credential/         # Folder untuk kredensial (tidak ada di repository)
    └── credential.json # File kredensial dari Google Cloud (harus ditambahkan secara manual)
```

## Persiapan dan Instalasi

### 1. Clone Repository
```sh
git clone https://github.com/mzkki25/planktonsee.git
cd repository
```

### 2. Buat dan Aktifkan Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # Untuk MacOS/Linux
venv\Scripts\activate     # Untuk Windows
```

### 3. Install Dependensi
```sh
pip install -r requirements.txt
```

### 4. Tambahkan File `credential.json`
Buat folder `credential/` dan letakkan file `credential.json` dari Google Cloud di dalamnya.

```
mkdir credential
mv path/to/your/credential.json credential/
```

### 5. Buat File `.env`
Buat file `.env` di root folder untuk menyimpan API key dan secret key lainnya.

```
GOOGLE_API_KEY="your_google_api_key_here"
```

### 6. Jalankan Aplikasi
```sh
python main.py
```
Aplikasi akan berjalan di `http://127.0.0.1:5000/`

## Deployment
Aplikasi ini dapat dideploy ke Google Cloud Run atau Railway dengan konfigurasi yang telah disediakan dalam `Dockerfile` dan `railway.json`. Pastikan untuk mengganti kredensial dan konfigurasi yang sesuai sebelum melakukan deployment.

