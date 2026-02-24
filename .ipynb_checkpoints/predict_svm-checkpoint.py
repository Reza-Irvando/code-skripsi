import os
import rasterio
import numpy as np
import joblib
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from tqdm import tqdm

# ================= 1. KONFIGURASI PENGGUNA =================
DIR_ORTO = "eksport-orto"
DIR_CLASS = "hasil-ndvi-class"
DIR_MODEL = "model_kombinasi"
DIR_OUTPUT = "hasil_prediksi_final"
CHUNK_SIZE = 100000 

SEMUA_TANGGAL = ["20230814", "20230828", "20230906", "20230919", "20231005"]

# -----------------------------------------------------------
# SILAKAN GANTI 2 TANGGAL INI SECARA MANUAL SESUAI KEINGINAN
TANGGAL_UJI_PILIHAN = ["20230906", "20231005"] 
# -----------------------------------------------------------

if not os.path.exists(DIR_OUTPUT):
    os.makedirs(DIR_OUTPUT)

# ================= 2. AUTO-DETECT MODEL =================
# Mencari 3 tanggal sisanya untuk menentukan nama model yang tepat
tanggal_latih = sorted([t for t in SEMUA_TANGGAL if t not in TANGGAL_UJI_PILIHAN])
nama_file_model = f"model_svm_{tanggal_latih[0]}_{tanggal_latih[1]}_{tanggal_latih[2]}.pkl"
path_model = os.path.join(DIR_MODEL, nama_file_model)

print("--- KONFIGURASI CROSS-VALIDATION ---")
print(f"Data Uji Manual  : {TANGGAL_UJI_PILIHAN}")
print(f"Model Terdeteksi : Dilatih dari {tanggal_latih}")

if not os.path.exists(path_model):
    raise FileNotFoundError(f"Model {nama_file_model} tidak ditemukan. Jalankan script training dulu.")

print(f"Memuat model {nama_file_model}...")
model_svm = joblib.load(path_model)

# ================= 3. PROSES PREDIKSI UNTUK MASING-MASING CITRA UJI =================
for tanggal_uji in TANGGAL_UJI_PILIHAN:
    print(f"\n{'='*50}")
    print(f"MEMPROSES CITRA UJI: {tanggal_uji}")
    print(f"{'='*50}")
    
    path_orto_uji = os.path.join(DIR_ORTO, f"{tanggal_uji}.tif")
    path_class_uji = os.path.join(DIR_CLASS, f"class-{tanggal_uji}.tif")
    
    with rasterio.open(path_orto_uji) as src_orto, rasterio.open(path_class_uji) as src_class:
        orto_meta = src_orto.meta.copy()
        img_orto = src_orto.read()
        
        # ALIGNMENT GEOSPASIAL (Memastikan posisi pas 100%)
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

    # PREDIKSI
    bands, h, w = orto_cropped.shape
    orto_flat = orto_cropped.reshape(bands, h * w).T
    ref_flat = ref_cropped.flatten()
    
    mask_valid = np.isin(ref_flat, [1, 2, 3])
    X_test = orto_flat[mask_valid]
    y_true = ref_flat[mask_valid]

    y_pred_list = []
    for i in tqdm(range(0, len(X_test), CHUNK_SIZE), desc=f"Prediksi {tanggal_uji}"):
        chunk = X_test[i : i + CHUNK_SIZE]
        y_pred_list.append(model_svm.predict(chunk))
    
    y_pred_valid = np.concatenate(y_pred_list)
    
    pred_flat = np.zeros(h * w, dtype=np.uint8)
    pred_flat[mask_valid] = y_pred_valid
    pred_cropped = pred_flat.reshape(h, w)

    # EVALUASI & EXPORT
    cm = confusion_matrix(y_true, y_pred_valid, labels=[1, 2, 3])
    oa = accuracy_score(y_true, y_pred_valid)
    kappa = cohen_kappa_score(y_true, y_pred_valid)

    print(f"\n--- HASIL AKURASI: {tanggal_uji} ---")
    print(f"Overall Accuracy : {oa * 100:.2f}%")
    print(f"Kappa Score      : {kappa:.4f}")
    
    meta_export = orto_meta.copy()
    meta_export.update({'height': crop_height, 'width': crop_width, 'transform': crop_transform, 'count': 1, 'dtype': 'uint8', 'nodata': 0})
    
    path_out_pred = os.path.join(DIR_OUTPUT, f"prediksi_{tanggal_uji}.tif")
    path_out_ref = os.path.join(DIR_OUTPUT, f"referensi_{tanggal_uji}.tif")
    
    with rasterio.open(path_out_pred, 'w', **meta_export) as dst: dst.write(pred_cropped, 1)
    with rasterio.open(path_out_ref, 'w', **meta_export) as dst: dst.write(ref_cropped, 1)

print("\nSeluruh proses uji silang selesai!")