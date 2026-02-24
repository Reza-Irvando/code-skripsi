import os
import sys
import time
import rasterio
import numpy as np
import joblib
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from tqdm import tqdm
import itertools

# IMPOR cuML
from cuml.svm import SVC

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("Laporan_Proses_Prediksi_CUDA.txt")

DIR_ORTO = "eksport-orto"
DIR_CLASS = "hasil-ndvi-class"
DIR_MODEL = "model_kombinasi"
DIR_OUTPUT = "hasil_prediksi_batch"

# CHUNK SIZE DITINGKATKAN KARENA MENGGUNAKAN GPU
CHUNK_SIZE = 500000 

SEMUA_TANGGAL = ["20230814", "20230828", "20230906", "20230919", "20231005"]
kombinasi_latih = list(itertools.combinations(SEMUA_TANGGAL, 3))

if not os.path.exists(DIR_OUTPUT):
    os.makedirs(DIR_OUTPUT)

print("\n" + "="*65)
print("  TAHAP 0: VERIFIKASI DATASET & MODEL")
print("=================================================================")
ada_error = False

for d in [DIR_ORTO, DIR_CLASS, DIR_MODEL]:
    if not os.path.exists(d):
        print(f" [X] ERROR: Direktori '{d}' tidak ditemukan.")
        ada_error = True

if not ada_error:
    for tgl in SEMUA_TANGGAL:
        if not os.path.exists(os.path.join(DIR_ORTO, f"{tgl}.tif")) or not os.path.exists(os.path.join(DIR_CLASS, f"class-{tgl}.tif")):
            print(f" [X] ERROR: File citra untuk tanggal {tgl} hilang.")
            ada_error = True
            
    for combo in kombinasi_latih:
        if not os.path.exists(os.path.join(DIR_MODEL, f"model_svm_{combo[0]}_{combo[1]}_{combo[2]}.pkl")):
            print(f" [X] ERROR: Model untuk kombinasi {combo} hilang.")
            ada_error = True

if ada_error:
    print("\n[!] VERIFIKASI GAGAL. Program dihentikan.")
    sys.exit(1)
else:
    print(" [V] STATUS AMAN: Seluruh file lengkap!")

print("\n=================================================================")
print("  TAHAP 1: PREDIKSI & GENERATE CITRA (CUDA ACCELERATED)")
print("=================================================================")
waktu_mulai_total = time.time()

for idx, combo in enumerate(kombinasi_latih, 1):
    waktu_model_mulai = time.time()
    tgl_latih = list(combo)
    tgl_uji = [t for t in SEMUA_TANGGAL if t not in tgl_latih]
    
    kode_model = f"{tgl_latih[0][-4:]}_{tgl_latih[1][-4:]}_{tgl_latih[2][-4:]}"
    path_model = os.path.join(DIR_MODEL, f"model_svm_{tgl_latih[0]}_{tgl_latih[1]}_{tgl_latih[2]}.pkl")
    
    print(f"\n\n{'='*65}")
    print(f" MENGEKSEKUSI MODEL [{idx}/10] : {kode_model}")
    print(f"{'='*65}")

    model_svm = joblib.load(path_model)
    
    for uji in tgl_uji:
        print(f"\n-> Proses Citra Uji: {uji}")
        
        nama_file_pred = f"pred_Model_{kode_model}_Uji_{uji}.tif"
        nama_file_ref = f"ref_Model_{kode_model}_Uji_{uji}.tif"
        path_out_pred = os.path.join(DIR_OUTPUT, nama_file_pred)
        path_out_ref = os.path.join(DIR_OUTPUT, nama_file_ref)
        
        if os.path.exists(path_out_pred) and os.path.exists(path_out_ref):
            print(f"   [INFO] Citra uji '{uji}' sudah diprediksi sebelumnya. Skip.")
            continue 

        with rasterio.open(os.path.join(DIR_ORTO, f"{uji}.tif")) as src_orto, rasterio.open(os.path.join(DIR_CLASS, f"class-{uji}.tif")) as src_class:
            orto_meta = src_orto.meta.copy()
            img_orto = src_orto.read()
            
            img_ref_aligned = np.zeros((src_orto.height, src_orto.width), dtype=np.uint8)
            reproject(
                source=rasterio.band(src_class, 1),
                destination=img_ref_aligned,
                src_transform=src_class.transform,
                src_crs=src_class.crs,
                dst_transform=src_orto.transform,
                dst_crs=src_orto.crs,
                resampling=Resampling.nearest
            )

            valid_y, valid_x = np.where((img_ref_aligned == 1) | (img_ref_aligned == 2) | (img_ref_aligned == 3))
            min_x, max_x = valid_x.min(), valid_x.max()
            min_y, max_y = valid_y.min(), valid_y.max()
            
            crop_width = (max_x - min_x) + 1
            crop_height = (max_y - min_y) + 1
            
            orto_cropped = img_orto[:, min_y:max_y+1, min_x:max_x+1]
            ref_cropped = img_ref_aligned[min_y:max_y+1, min_x:max_x+1]
            crop_window = Window(min_x, min_y, crop_width, crop_height)
            crop_transform = src_orto.window_transform(crop_window)

        bands, h, w = orto_cropped.shape
        orto_flat = orto_cropped.reshape(bands, h * w).T
        ref_flat = ref_cropped.flatten()
        
        mask_valid = np.isin(ref_flat, [1, 2, 3])
        X_test = orto_flat[mask_valid]
        print(f"   Total Piksel Lahan Uji : {len(X_test):,} piksel")
        
        waktu_pred_mulai = time.time()
        y_pred_list = []
        
        # PREDIKSI DI GPU (Membutuhkan float32)
        for i in tqdm(range(0, len(X_test), CHUNK_SIZE), desc=f"   Prediksi (GPU)", unit="chunk"):
            chunk = X_test[i : i + CHUNK_SIZE].astype(np.float32)
            y_pred_list.append(model_svm.predict(chunk))
            
        y_pred_valid = np.concatenate(y_pred_list)
        waktu_tempuh_citra = time.time() - waktu_pred_mulai
        print(f"   Waktu Prediksi Citra   : {waktu_tempuh_citra:.2f} detik")

        pred_flat = np.zeros(h * w, dtype=np.uint8)
        pred_flat[mask_valid] = y_pred_valid
        pred_cropped = pred_flat.reshape(h, w)
        
        meta_export = orto_meta.copy()
        meta_export.update({'height': crop_height, 'width': crop_width, 'transform': crop_transform, 'count': 1, 'dtype': 'uint8', 'nodata': 0})
        
        with rasterio.open(path_out_pred, 'w', **meta_export) as dst: dst.write(pred_cropped, 1)
        with rasterio.open(path_out_ref, 'w', **meta_export) as dst: dst.write(ref_cropped, 1)

    waktu_model_tempuh = time.time() - waktu_model_mulai
    menit = int(waktu_model_tempuh // 60)
    detik = waktu_model_tempuh % 60
    print(f"\n   --------------------------------------------------------------")
    print(f"   TOTAL WAKTU MODEL [{idx}/10] (Sesi Ini): {menit} menit {detik:.2f} detik")
    print(f"   --------------------------------------------------------------")

total_waktu_detik = time.time() - waktu_mulai_total
jam = int(total_waktu_detik // 3600)
menit = int((total_waktu_detik % 3600) // 60)
detik = total_waktu_detik % 60

print("\n" + "="*65)
print(f"TAHAP 1 SELESAI! Waktu Keseluruhan Sesi Ini: {jam} jam, {menit} menit, {detik:.2f} detik.")
print("="*65)