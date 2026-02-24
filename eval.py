import os
import sys
import rasterio
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import itertools

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("Laporan_Akurasi_10_Model_Final.txt")

DIR_OUTPUT = "hasil_prediksi_batch" 
SEMUA_TANGGAL = ["20230814", "20230828", "20230906", "20230919", "20231005"]
kombinasi_latih = list(itertools.combinations(SEMUA_TANGGAL, 3))

print("=================================================================")
print("  TAHAP 2: EVALUASI AKURASI TOTAL (LEAVE-P-OUT CROSS VALIDATION)")
print("=================================================================")

list_oa = []
list_kappa = []

for idx, combo in enumerate(kombinasi_latih, 1):
    tgl_latih = list(combo)
    tgl_uji = [t for t in SEMUA_TANGGAL if t not in tgl_latih]
    kode_model = f"{tgl_latih[0][-4:]}_{tgl_latih[1][-4:]}_{tgl_latih[2][-4:]}"
    
    print(f"\n\n{'='*65}")
    print(f" EVALUASI MODEL [{idx}/10] : {kode_model}")
    print(f"{'='*65}")

    y_true_gabungan = []
    y_pred_gabungan = []

    for uji in tgl_uji:
        path_pred = os.path.join(DIR_OUTPUT, f"pred_Model_{kode_model}_Uji_{uji}.tif")
        path_ref = os.path.join(DIR_OUTPUT, f"ref_Model_{kode_model}_Uji_{uji}.tif")
        
        if not os.path.exists(path_pred) or not os.path.exists(path_ref):
            print(f"   [Skip] File prediksi/referensi untuk {uji} tidak ditemukan.")
            continue
            
        with rasterio.open(path_pred) as src_pred, rasterio.open(path_ref) as src_ref:
            pred_data = src_pred.read(1).flatten()
            ref_data = src_ref.read(1).flatten()
            
            mask_valid = np.isin(ref_data, [1, 2, 3])
            y_pred_gabungan.extend(pred_data[mask_valid])
            y_true_gabungan.extend(ref_data[mask_valid])

    if len(y_true_gabungan) == 0:
        continue

    y_true_all = np.array(y_true_gabungan)
    y_pred_all = np.array(y_pred_gabungan)
    
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[1, 2, 3])
    oa = accuracy_score(y_true_all, y_pred_all)
    kappa = cohen_kappa_score(y_true_all, y_pred_all)
    
    list_oa.append(oa)
    list_kappa.append(kappa)

    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)

    print(f"   Total Piksel Dievaluasi: {len(y_true_all):,} piksel")
    print("\nConfusion Matrix (Baris: Reference, Kolom: Prediction):")
    print("             | Pred_1 | Pred_2 | Pred_3 || Row Sum")
    print("--------------------------------------------------")
    print(f"Ref_1 (Ren)  | {cm[0,0]:6d} | {cm[0,1]:6d} | {cm[0,2]:6d} || {row_sums[0]:7d}")
    print(f"Ref_2 (Sed)  | {cm[1,0]:6d} | {cm[1,1]:6d} | {cm[1,2]:6d} || {row_sums[1]:7d}")
    print(f"Ref_3 (Tig)  | {cm[2,0]:6d} | {cm[2,1]:6d} | {cm[2,2]:6d} || {row_sums[2]:7d}")
    print("--------------------------------------------------")
    print(f"Col Sum      | {col_sums[0]:6d} | {col_sums[1]:6d} | {col_sums[2]:6d} || {np.sum(cm):7d}")

    print("\nProducer's Accuracy (PA) & User's Accuracy (UA):")
    kelas_label = ["Kelas 1 (Rendah)", "Kelas 2 (Sedang)", "Kelas 3 (Tinggi)"]
    for i in range(3):
        pa = (cm[i, i] / row_sums[i] * 100) if row_sums[i] > 0 else 0
        ua = (cm[i, i] / col_sums[i] * 100) if col_sums[i] > 0 else 0
        print(f" - {kelas_label[i]:16} -> PA: {pa:6.2f}%  |  UA: {ua:6.2f}%")

    print(f"\nOverall Accuracy (OA) : {oa * 100:.2f}%")
    print(f"Kappa Coefficient     : {kappa:.4f}")

print("\n\n" + "#"*65)
print(" RANGKUMAN KESELURUHAN (GRAND AVERAGE DARI 10 KOMBINASI MODEL)")
print("#"*65)
if len(list_oa) > 0:
    print(f"Rata-rata Overall Accuracy : {np.mean(list_oa) * 100:.2f}%")
    print(f"Rata-rata Kappa Coefficient: {np.mean(list_kappa):.4f}")
print("#"*65)