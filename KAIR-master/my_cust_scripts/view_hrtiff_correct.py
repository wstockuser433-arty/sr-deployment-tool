import rasterio
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# CHANGE ONLY THIS
# ==================================================
# image_path = r"/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/airbus/IMG_PNEO3_STD_202601250555400_PMS-FS_ORT_1e224074-3a2a-4f29-cfe4-ab0a484fbd47_NED_R1C1.JP2"
# image_path = r"/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/airbus/IMG_PNEO3_STD_202601250555400_PMS-FS_ORT_1e224074-3a2a-4f29-cfe4-ab0a484fbd47_RGB_R1C1.JP2"
# image_path = r"/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/maxar-samples/Rome_Colosseum_2022-03-22_WV03_HD/050012575010_01/050012575010_01_P001_PSH/22MAR22095810-S2AS-050012575010_01_P001.TIF"
# image_path = r"/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/maxar-samples/Barcelona_2022-04-01_WV03_HD/050012560010_01/050012560010_01_P001_PSH/22APR01105255-S3DS_R1C1-050012560010_01_P001.TIF"
image_path = r"/home/hassan/Documents/Hassan/SR-Enhancement-Work/Datasets/maxar-dataset-pak/pakistan-flood-images-36cm/1040010076369100-visual.tif"
# ==================================================

# --------------------------------------------------
# OPEN IMAGE
# --------------------------------------------------

with rasterio.open(image_path) as src:

    print("=" * 80)
    print("IMAGE INFORMATION")
    print("=" * 80)

    print(f"File            : {image_path}")
    print(f"Width           : {src.width}")
    print(f"Height          : {src.height}")
    print(f"Bands           : {src.count}")
    print(f"Data Type       : {src.dtypes}")
    print(f"CRS             : {src.crs}")
    print(f"NoData          : {src.nodata}")
    print(f"ColorInterp     : {src.colorinterp}")

    print("\nAffine Transform")
    print(src.transform)

    print("\n" + "=" * 80)
    print("BAND INFORMATION")
    print("=" * 80)

    print(f"Total Bands     : {src.count}")

    for band_idx in range(1, src.count + 1):
        tags = src.tags(band_idx)

        band_name = (
            src.descriptions[band_idx - 1]
            or tags.get("BAND_NAME")
            or tags.get("band_name")
            or tags.get("NAME")
            or tags.get("name")
            or f"Band {band_idx}"
        )

        band_description = (
            tags.get("DESCRIPTION")
            or tags.get("description")
            or tags.get("DESC")
            or tags.get("desc")
            or "No description available"
        )

        print(f"Band {band_idx}: name='{band_name}', description='{band_description}'")



    print("\n" + "=" * 80)
    print("BAND STATISTICS")
    print("=" * 80)

    for band_idx in range(1, src.count + 1):

        band = src.read(band_idx)

        print(
            f"Band {band_idx}: "
            f"min: {band.min():>8} "
            f"max: {band.max():>8} "
            f"mean: {band.mean():>10.2f} "
            f"std: {band.std():>10.2f}"
        )

        print(
            f"P1: {np.percentile(band, 1):.0f}  "
            f"P5: {np.percentile(band, 5):.0f}  "
            f"P95: {np.percentile(band, 95):.0f}  "
            f"P99: {np.percentile(band, 99):.0f}  "
            f"P99.5: {np.percentile(band, 99.5):.0f}  "
            f"P99.9: {np.percentile(band, 99.9):.0f}"
        )


    # --------------------------------------------------
    # Read RGB bands 
    # --------------------------------------------------

    # rgb = src.read([1, 2, 3])
    scale_factor = 8

    rgb = src.read(
        [1, 2, 3]            
,
        out_shape=(
            3,
            src.height // scale_factor,
            src.width // scale_factor
        )
    )

# --------------------------------------------------
# PREPARE RGB IMAGE
# --------------------------------------------------

rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)

# --------------------------------------------------
# GLOBAL PERCENTILE STRETCH
# (Preserves color balance)
# --------------------------------------------------

low = np.percentile(rgb, 2)
high = np.percentile(rgb, 98)

rgb_display = (rgb - low) / (high - low)
rgb_display = np.clip(rgb_display, 0, 1)

print("\n" + "=" * 80)
print("DISPLAY STRETCH")
print("=" * 80)

print(f"Global P2  : {low:.2f}")
print(f"Global P98 : {high:.2f}")

# --------------------------------------------------
# DISPLAY IMAGE
# --------------------------------------------------

plt.figure(figsize=(12, 12))
plt.imshow(rgb_display)
plt.title("RGB Preview (Global 2-98% Stretch)")
plt.axis("off")
plt.show()

# --------------------------------------------------
# RGB HISTOGRAMS
# --------------------------------------------------

plt.figure(figsize=(12, 6))

for i, label in enumerate(["Red", "Green", "Blue"]):
    plt.hist(
        rgb[:, :, i].ravel(),
        bins=256,
        alpha=0.4,
        label=label
    )

plt.title("RGB Histograms")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# --------------------------------------------------
# CHANNEL MEANS SUMMARY
# --------------------------------------------------

print("\n" + "=" * 80)
print("CHANNEL MEANS")
print("=" * 80)

print(f"Red Mean   : {rgb[:,:,0].mean():.2f}")
print(f"Green Mean : {rgb[:,:,1].mean():.2f}")
print(f"Blue Mean  : {rgb[:,:,2].mean():.2f}")