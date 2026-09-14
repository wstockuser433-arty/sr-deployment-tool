"""
SR Quality Assessment Code Cells
=================================
Copy and paste these into separate code cells in your Jupyter notebook:
image_previewer.ipynb

Each section is clearly marked as CELL 1, 2, 3, 4
"""

# ============================================================================
# CELL 1: CROP PATCH COMPARISON (Full Resolution, No Downsampling)
# ============================================================================
"""
Copy this entire block into a new code cell in the notebook
"""

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import cv2

# Paths
lr_path = r"/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/airbus/outputs/lr_x2/IMG_PNEO3_STD_202601250555400_PMS-FS_ORT_1e224074-3a2a-4f29-cfe4-ab0a484fbd47_RGB_R1C1.png"
sr_path = r"/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/airbus/outputs/sr_x2/IMG_PNEO3_STD_202601250555400_PMS-FS_ORT_1e224074-3a2a-4f29-cfe4-ab0a484fbd47_RGB_R1C1_SwinIR.png"
hr_path = r"/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/airbus/outputs/hr_processed/IMG_PNEO3_STD_202601250555400_PMS-FS_ORT_1e224074-3a2a-4f29-cfe4-ab0a484fbd47_RGB_R1C1.png"

# Load images
lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

print(f"LR shape: {lr_img.shape}")
print(f"SR shape: {sr_img.shape}")
print(f"HR shape: {hr_img.shape}")

# Extract crop from center (interesting region with detail)
crop_size = 256  # 256×256 patch at FULL RESOLUTION (no downsampling)

# Center coordinates
sr_h, sr_w = sr_img.shape[:2]
cx, cy = sr_w // 2, sr_h // 2
x1 = max(0, cx - crop_size // 2)
x2 = min(sr_w, x1 + crop_size)
y1 = max(0, cy - crop_size // 2)
y2 = min(sr_h, y1 + crop_size)

sr_crop = sr_img[y1:y2, x1:x2]
hr_crop = hr_img[y1:y2, x1:x2]

# LR crop (at LR resolution)
lr_y1 = y1 // 2
lr_y2 = y2 // 2
lr_x1 = x1 // 2
lr_x2 = x2 // 2
lr_crop = lr_img[lr_y1:lr_y2, lr_x1:lr_x2]

print(f"\n{'='*70}")
print(f"Cropped regions (center patch, {crop_size}×{crop_size} SR space)")
print(f"{'='*70}")
print(f"LR crop shape: {lr_crop.shape}")
print(f"SR crop shape: {sr_crop.shape}")
print(f"HR crop shape: {hr_crop.shape}")

# Calculate metrics on crop
psnr_sr_hr = psnr(hr_crop, sr_crop, data_range=1.0)
ssim_sr_hr = ssim(hr_crop, sr_crop, data_range=1.0, channel_axis=2)

# Bicubic baseline: Upsample LR with bicubic
lr_bicubic = cv2.resize(lr_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
lr_bic_crop = lr_bicubic[y1:y2, x1:x2]
psnr_bic_hr = psnr(hr_crop, lr_bic_crop, data_range=1.0)
ssim_bic_hr = ssim(hr_crop, lr_bic_crop, data_range=1.0, channel_axis=2)

print(f"\nMetrics vs. HR (center crop):")
print(f"  SR-PSNR:      {psnr_sr_hr:.2f} dB")
print(f"  SR-SSIM:      {ssim_sr_hr:.4f}")
print(f"  Bicubic-PSNR: {psnr_bic_hr:.2f} dB  (baseline)")
print(f"  Bicubic-SSIM: {ssim_bic_hr:.4f}  (baseline)")
print(f"  PSNR gain:    {psnr_sr_hr - psnr_bic_hr:+.2f} dB  {'✓ SR better!' if psnr_sr_hr > psnr_bic_hr else '✗ Worse'}")

# DISPLAY: LR (upsampled) vs SR vs HR
lr_upsampled = cv2.resize(lr_crop, (sr_crop.shape[1], sr_crop.shape[0]), interpolation=cv2.INTER_CUBIC)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Convert BGR→RGB for display
axes[0].imshow(cv2.cvtColor((lr_upsampled * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
axes[0].set_title(f"LR (Bicubic)\nPSNR={psnr_bic_hr:.2f} dB")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor((sr_crop * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
axes[1].set_title(f"SR (SwinIR)\nPSNR={psnr_sr_hr:.2f} dB")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor((hr_crop * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
axes[2].set_title(f"HR (Reference)")
axes[2].axis("off")

plt.suptitle(f"Center Crop Comparison (256×256 @full resolution)\nGain: {psnr_sr_hr - psnr_bic_hr:+.2f} dB", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ============================================================================
# CELL 2: FREQUENCY ANALYSIS (Fourier Domain)
# ============================================================================
"""
Copy this entire block into a new code cell in the notebook
(This cell depends on variables from CELL 1)
"""

def get_frequency_spectrum(img_crop, title=""):
    """Compute power spectrum in frequency domain"""
    # Convert to grayscale for FFT
    if img_crop.ndim == 3:
        gray = cv2.cvtColor((img_crop * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img_crop.astype(np.float32)
    
    # FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    # Log scale for better visualization
    log_mag = np.log1p(magnitude)
    
    return log_mag

# Compute spectra
spectrum_lr = get_frequency_spectrum(lr_upsampled, "LR")
spectrum_sr = get_frequency_spectrum(sr_crop, "SR")
spectrum_hr = get_frequency_spectrum(hr_crop, "HR")

# DISPLAY: Frequency spectra comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

im0 = axes[0].imshow(spectrum_lr, cmap="hot")
axes[0].set_title("LR (Bicubic)\n[Mostly low-freq]")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(spectrum_sr, cmap="hot")
axes[1].set_title("SR (SwinIR)\n[Should have high-freq energy]")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(spectrum_hr, cmap="hot")
axes[2].set_title("HR (Reference)\n[Full spectrum]")
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle("Frequency Domain Analysis (Fourier Log-Magnitude)\nIf SR works: SR spectrum should be closer to HR than LR", 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Radial frequency profile (Distance from center = frequency)
def compute_radial_spectrum(log_mag):
    """Average power at each radial frequency distance"""
    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((y - cy)**2 + (x - cx)**2).astype(int)
    
    radii = np.arange(0, min(h, w) // 2)
    radial_profile = []
    
    for r in radii:
        mask = distance == r
        if mask.sum() > 0:
            radial_profile.append(log_mag[mask].mean())
        else:
            radial_profile.append(0)
    
    return np.array(radial_profile)

profile_lr = compute_radial_spectrum(spectrum_lr)
profile_sr = compute_radial_spectrum(spectrum_sr)
profile_hr = compute_radial_spectrum(spectrum_hr)

# Plot radial profiles
plt.figure(figsize=(12, 5))
radii = np.arange(len(profile_lr))

plt.plot(radii, profile_lr, 'b-', linewidth=2, label="LR (Bicubic)", alpha=0.7)
plt.plot(radii, profile_sr, 'g-', linewidth=2, label="SR (SwinIR)", alpha=0.7)
plt.plot(radii, profile_hr, 'r--', linewidth=2, label="HR (Reference)", alpha=0.7)

plt.xlabel("Radial Frequency (pixels from center)", fontsize=11)
plt.ylabel("Log-Magnitude", fontsize=11)
plt.title("Radial Frequency Profile\n(Higher curve = more high-frequency energy recovered)", fontsize=12, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nFrequency Analysis Summary:")
print(f"  High-freq (outer 20%):  LR={profile_lr[-20:].mean():.2f}  SR={profile_sr[-20:].mean():.2f}  HR={profile_hr[-20:].mean():.2f}")
print(f"  {'✓ SR recovering high-freq!' if profile_sr[-20:].mean() > profile_lr[-20:].mean() else '✗ Low-frequency boost only'}")


# ============================================================================
# CELL 3: DIFFERENCE MAPS (What did SR add?)
# ============================================================================
"""
Copy this entire block into a new code cell in the notebook
(This cell depends on variables from CELL 1)
"""

# Compute error maps (absolute difference from HR)
err_lr = np.abs(lr_upsampled - hr_crop).mean(axis=2)
err_sr = np.abs(sr_crop - hr_crop).mean(axis=2)

# DISPLAY: Error maps
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im0 = axes[0].imshow(err_lr, cmap="hot")
axes[0].set_title(f"LR Error Map\nMean Abs Diff: {err_lr.mean():.4f}")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], label="Error", fraction=0.046, pad=0.04)

im1 = axes[1].imshow(err_sr, cmap="hot")
axes[1].set_title(f"SR Error Map\nMean Abs Diff: {err_sr.mean():.4f}")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], label="Error", fraction=0.046, pad=0.04)

plt.suptitle("Residual Error vs. HR Reference\n(Darker = closer to HR)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\nError Statistics:")
print(f"  LR Mean Error:  {err_lr.mean():.4f}  (Std: {err_lr.std():.4f})")
print(f"  SR Mean Error:  {err_sr.mean():.4f}  (Std: {err_sr.std():.4f})")
print(f"  Error Reduction: {(err_lr.mean() - err_sr.mean()) / err_lr.mean() * 100:.1f}%")
print(f"  {'✓ SR reduced error!' if err_sr.mean() < err_lr.mean() else '✗ No improvement'}")

# DISPLAY: Residual image (SR - HR) to see hallucinations
residual_sr = sr_crop - hr_crop
residual_lr = lr_upsampled - hr_crop

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residual with centered colormap
vmax = max(np.abs(residual_lr).max(), np.abs(residual_sr).max())

im0 = axes[0].imshow(residual_lr.mean(axis=2), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes[0].set_title("LR - HR (Bicubic residual)")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(residual_sr.mean(axis=2), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes[1].set_title("SR - HR (SwinIR residual)")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.suptitle("Residual Maps (Red=Bright artifacts, Blue=Dark artifacts)\nSR residual should be smaller & less structured", 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()


# ============================================================================
# CELL 4: INTERPRETATION GUIDE (Pass/Fail Checklist)
# ============================================================================
"""
Copy this entire block into a new code cell in the notebook
"""

interpretation = """
╔════════════════════════════════════════════════════════════════════════╗
║                    SR QUALITY ASSESSMENT GUIDE                         ║
╚════════════════════════════════════════════════════════════════════════╝

1. METRIC COMPARISON (vs. HR reference)
   ├─ PSNR (Peak Signal-to-Noise Ratio)
   │  ├─ Higher is better (typical: 25-35 dB)
   │  ├─ SR-PSNR > Bicubic-PSNR → ✓ SR is sharpening
   │  └─ Gain of 2-3 dB suggests good restoration
   │
   ├─ SSIM (Structural Similarity Index)
   │  ├─ Range: 0 to 1 (1 = perfect)
   │  ├─ Typical: 0.7-0.95
   │  └─ SR-SSIM > Bicubic-SSIM → ✓ Better structure preservation
   │
   └─ Visual Inspection
      ├─ LR (Bicubic): Blurry, soft edges
      ├─ SR (SwinIR):  Should have crisp edges, recovered texture
      └─ HR (Reference): Ground truth

2. FREQUENCY ANALYSIS
   ├─ LR spectrum: Energy concentrated at LOW frequencies (blurry)
   ├─ HR spectrum: Energy spread across ALL frequencies (sharp)
   └─ SR spectrum: Should be CLOSER to HR than LR
      └─ If SR ≈ LR → ✗ Model not working
      └─ If SR ≈ HR → ✓ Good restoration

3. ERROR MAPS
   ├─ Darker regions = closer to HR (better)
   ├─ SR error < LR error → ✓ Model improving
   ├─ Structured patterns → ✗ Model hallucinating
   └─ Random noise patterns → ✓ Normal (harder pixels)

4. RESIDUAL ANALYSIS
   ├─ Residual = Output - HR Reference
   ├─ Red regions: Output too bright → Artifacts
   ├─ Blue regions: Output too dark → Artifacts
   ├─ Gray/neutral: Close to HR ✓
   └─ SR residual should be smaller than LR residual

═══════════════════════════════════════════════════════════════════════════

✓ YOUR SR IS WORKING if:
  • PSNR gain ≥ 1.5 dB over bicubic
  • SSIM ≥ 0.85 (very good structural match)
  • Frequency spectrum similar to HR (especially high-freq boost)
  • Error maps show darker regions for SR
  • Visual inspection shows crisp edges and recovered details

⚠ WARNING SIGNS (model not working):
  • PSNR loss compared to bicubic
  • SSIM < 0.7 or degrading
  • SR spectrum identical to LR (just blurry copy)
  • Colored artifacts in residuals
  • Heavy noise amplification

═══════════════════════════════════════════════════════════════════════════

NEXT STEPS:
1. Sample multiple crops from different regions (urban, rural, water)
2. Test at different upsampling factors (2×, 3×, 4×)
3. Compare with other baselines (ESRGAN, Real-ESRGAN)
4. Check log file for average metrics across entire dataset
"""

print(interpretation)
