import os
import time
import rasterio
import numpy as np
import joblib
from rasterio.windows import Window
from tqdm import tqdm
import itertools

# IMPOR cuML UNTUK GPU ACCELERATION
from cuml.svm import SVC 

# ================= 1. KONFIGURASI UTAMA =================
DIR_ORTO = "eksport-orto"
DIR_CLASS = "hasil-ndvi-class"
DIR_MODEL = "model_kombinasi"

SEMUA_TANGGAL = ["20230814", "20230828", "20230906", "20230919", "20231005"]

# BATASAN SAMPEL DITINGKATKAN MENJADI 100.000 PER KELAS
# Total per citra = 150.000 piksel. Total per model (3 citra) = 450.000 piksel latih.
SAMPEL_PER_KELAS = 50000  

if not os.path.exists(DIR_MODEL):
    os.makedirs(DIR_MODEL)

print("=================================================================")
print(f"  TAHAP 1: EKSTRAKSI SAMPEL CITRA ({SAMPEL_PER_KELAS:,} per kelas)")
print("=================================================================")
data_cache = {}

for tgl in tqdm(SEMUA_TANGGAL, desc="Mengekstrak Data"):
    path_orto = os.path.join(DIR_ORTO, f"{tgl}.tif")
    path_class = os.path.join(DIR_CLASS, f"class-{tgl}.tif")
    
    with rasterio.open(path_orto) as src_orto, rasterio.open(path_class) as src_class:
        common_rows = min(src_orto.height, src_class.height)
        common_cols = min(src_orto.width, src_class.width)
        win = Window(0, 0, common_cols, common_rows)
        
        img_orto = src_orto.read(window=win)
        img_class = src_class.read(1, window=win)
        
        bands, rows, cols = img_orto.shape
        orto_flat = img_orto.reshape(bands, rows * cols).T
        class_flat = img_class.flatten()
        
        X_list_temp, y_list_temp = [], []
        
        for kelas in [1, 2, 3]:
            indeks_kelas = np.where(class_flat == kelas)[0]
            jumlah_ambil = min(SAMPEL_PER_KELAS, len(indeks_kelas))
            
            if jumlah_ambil > 0:
                indeks_terpilih = np.random.choice(indeks_kelas, size=jumlah_ambil, replace=False)
                X_list_temp.append(orto_flat[indeks_terpilih])
                y_list_temp.append(class_flat[indeks_terpilih])
                
        data_cache[tgl] = {
            'X': np.vstack(X_list_temp),
            'y': np.concatenate(y_list_temp)
        }

# ================= 2. GENERATE SELURUH KOMBINASI MODEL (DI GPU) =================
print("\n=================================================================")
print("  TAHAP 2: PELATIHAN 10 KOMBINASI MODEL (CUDA GPU)")
print("=================================================================")
kombinasi_latih = list(itertools.combinations(SEMUA_TANGGAL, 3))
total_model = len(kombinasi_latih)

# Mulai stopwatch global untuk proses training
waktu_mulai_training = time.time()

for idx, combo in enumerate(kombinasi_latih, 1):
    waktu_model_mulai = time.time()
    tgl_1, tgl_2, tgl_3 = combo
    nama_file_model = f"model_svm_{tgl_1}_{tgl_2}_{tgl_3}.pkl"
    path_model = os.path.join(DIR_MODEL, nama_file_model)
    
    print(f"\n[{idx}/{total_model}] Melatih Model: {tgl_1}, {tgl_2}, {tgl_3}")
    
    X_train = np.vstack([data_cache[tgl_1]['X'], data_cache[tgl_2]['X'], data_cache[tgl_3]['X']])
    y_train = np.concatenate([data_cache[tgl_1]['y'], data_cache[tgl_2]['y'], data_cache[tgl_3]['y']])
    
    # KONVERSI KE FLOAT32 UNTUK PERFORMA MAKSIMAL GPU
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    
    print(f"   -> Total Piksel Latih: {len(y_train):,} piksel")
    print(f"   -> Fitting SVM di GPU berjalan...")
    
    # Model SVC dari cuML
    model_svm = SVC(kernel='rbf', C=100, gamma='scale')
    model_svm.fit(X_train, y_train)
    
    joblib.dump(model_svm, path_model)
    
    waktu_tempuh_model = time.time() - waktu_model_mulai
    print(f"   -> SUKSES! Tersimpan: {nama_file_model} (Waktu: {waktu_tempuh_model:.2f} detik)")

    # ================= FITUR KALKULASI ETA GLOBAL =================
    sisa_model = total_model - idx
    if sisa_model > 0:
        waktu_berjalan_sesi = time.time() - waktu_mulai_training
        waktu_rata2_per_model = waktu_berjalan_sesi / idx
        estimasi_sisa_detik = waktu_rata2_per_model * sisa_model
        
        est_jam = int(estimasi_sisa_detik // 3600)
        est_menit = int((estimasi_sisa_detik % 3600) // 60)
        est_detik = estimasi_sisa_detik % 60
        
        print(f"   >> [ETA GLOBAL] Estimasi sisa waktu untuk {sisa_model} model berikutnya: {est_jam} Jam, {est_menit} Menit, {est_detik:.0f} Detik <<")
    # ==============================================================

# ================= 3. KALKULASI TOTAL WAKTU KESELURUHAN =================
total_waktu_detik = time.time() - waktu_mulai_training
jam = int(total_waktu_detik // 3600)
menit = int((total_waktu_detik % 3600) // 60)
detik = total_waktu_detik % 60

print("\n" + "="*65)
print(f"TAHAP 2 SELESAI! Seluruh 10 model berhasil dilatih.")
print(f"Waktu Keseluruhan Training: {jam} jam, {menit} menit, {detik:.2f} detik.")
print("="*65)