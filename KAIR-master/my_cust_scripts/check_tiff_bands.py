import rasterio
import numpy as np

path = "/home/hassan/Documents/Hassan/SR Enhancement Work/Datasets/WorldStrat/preprocessing_pipeline_test/set1/lr/Amnesty POI-2-3-3-1-L1C_data.tiff"

with rasterio.open(path) as src:
    print("Bands:", src.count)

    for i in range(1, src.count + 1):
        b = src.read(i)

        print(
            i,
            b.dtype,
            np.min(b),
            np.max(b),
            np.mean(b)
        )