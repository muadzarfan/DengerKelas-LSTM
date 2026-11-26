# DengerKelas — Transkripsi Bahasa Indonesia dengan LSTM

Muadz Arfan · Ilmu Komputer, Semester 5

## 1. Pendahuluan

DengerKelas dibuat untuk membantu mahasiswa mengikuti penjelasan di kelas lewat subtitle bahasa Indonesia. Di eksperimen ini, saya melatih model Long Short-Term Memory (LSTM) dari awal untuk melihat seberapa baik model kecil bisa mengubah rekaman pendek menjadi teks.

## 2. Deskripsi Kasus

Model menerima rekaman pendek dari satu orang yang berbicara dalam bahasa Indonesia. Hasilnya berupa transkrip dalam huruf kecil. Audio dibagi menjadi frame pendek, lalu setiap frame diubah menjadi 64 log-Mel features. Dua layer LSTM membaca frame secara berurutan dan memprediksi karakter.

Saya memakai Connectionist Temporal Classification (CTC) sebagai loss function karena dataset hanya punya transkrip per rekaman, tanpa timestamp untuk tiap karakter. Saat membuat prediksi, decoder memilih token dengan skor tertinggi di setiap frame, menggabungkan token yang sama dan berurutan, lalu membuang token blank.

## 3. Deskripsi Data

Saya memakai dataset FLEURS dari Google, khusus bagian bahasa Indonesia (`id_id`). Setiap contoh berisi rekaman mono 16 kHz dan transkripnya. Saya mengikuti pembagian train dan validation dari dataset dan memakai rekaman yang lolos pemeriksaan data proyek. Transkrip diubah ke huruf kecil, aksen dihilangkan, dan tanda baca diganti dengan spasi. Spasi yang berlebih dirapikan. [Dataset FLEURS](https://huggingface.co/datasets/google/fleurs)

| Split | Jumlah rekaman | Durasi audio | Kegunaan |
|---|---:|---:|---|
| Train | 2.538 | 8,88 jam | Melatih model dari audio dan transkrip |
| Validation | 345 | 1,14 jam | Mengukur kesalahan dan memilih checkpoint |

Rekaman FLEURS berisi kalimat yang dibacakan. Di kelas, orang biasanya berbicara lebih spontan dan ada suara latar yang bisa mengganggu.

Dataset FLEURS dari Google berlisensi [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Manifest di repo ini memuat metadata dan transkrip rekaman yang dipakai dalam eksperimen. Audio diambil dari arsip resmi saat persiapan data.

## 4. Analisis Performa

Saya melatih model selama 12 epoch di CPU laptop dengan AdamW, batch size 24, dan learning rate 0,002. Saya mengukur Word Error Rate (WER) dan Character Error Rate (CER) pada data validation. WER menghitung kesalahan pada tingkat kata dan CER pada tingkat karakter. Keduanya membagi jumlah penggantian, penghapusan, dan penambahan dengan jumlah kata atau karakter pada transkrip referensi. Nilai yang lebih kecil lebih baik. Spasi tidak dihitung dalam CER.

| Epoch | CTC loss saat training | WER validation | CER validation |
|---|---:|---:|---:|
| 1 | 3,462 | 100,00% | 100,00% |
| 5 | 2,512 | 100,00% | 72,23% |
| 12 (terbaik) | 1,369 | **97,95%** | **47,68%** |

Epoch 12 punya WER paling rendah. Training dan evaluasi memakan waktu sekitar 7 menit 25 detik. Dari hasilnya, model mulai mengenali bagian kata, tetapi masih sering salah eja atau salah menempatkan spasi. Pada contoh di bawah, `di atas` terbaca sebagai `diatas`.

| Transkrip referensi | Prediksi LSTM |
|---|---|
| `air tumpah di atas tanggul di bagian selebar 100 kaki` | `a ungpan diatas tan mudipadin sele balaratus pa ti` |

Dengan WER setinggi ini, hasil transkripsi masih sulit dibaca dan model belum layak dipakai untuk subtitle kelas. Semua angka di tabel berasal dari rekaman validation tanpa penambahan noise. [Hasil validasi lengkap](validation_results.json)

## 5. Kode dan Cara Menjalankan

Script training membaca audio, mengekstrak features, melatih LSTM, dan menghitung WER serta CER. Model punya 236.326 parameter. Arsitekturnya terdiri dari dua layer LSTM yang masing-masing punya 128 hidden units. Linear layer terakhir memberi skor untuk 37 karakter, termasuk spasi, dan satu token CTC blank. [Kode training](train.py)

Jalankan perintah berikut dari folder utama repo dengan Python 3.12 dan virtual environment yang sudah aktif.

```bash
python -m pip install -r requirements.txt
python train.py --prepare-data
python train.py
```

Perintah `--prepare-data` mengunduh arsip resmi train dan validation, sekitar 1,9 GB sebelum diekstrak. Setelah training, checkpoint terbaik dan prediksi tiap rekaman validation ada di folder `results/lstm/`. Untuk mengulang eksperimen dan menyimpan hasilnya di folder baru, tambahkan `--output results/lstm-repeat` pada perintah training.
