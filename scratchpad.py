# %%
import os
import urllib.request
import rasterio
import numpy as np

tile_url = (
    "https://minio.lab.sspcloud.fr/projet-funathon/2026/"
    "project3/data/images/LU000/"
    "2024/4042000_2951690_0_637.tif"
)

with rasterio.open(tile_url) as src:
    tile_crs = src.crs
    tile_bounds = src.bounds
    tile_count = src.count
    tile_height = src.height
    tile_width = src.width
    # Read RGB bands: B4 (Red), B3 (Green), B2 (Blue)
    rgb_data = src.read([4, 3, 2])

print(f"CRS:    {tile_crs}")
print(f"Bounds: {tile_bounds}")
print(f"Shape:  {tile_count} bands x {tile_height} x {tile_width} px")

# %%
import matplotlib.pyplot as plt

# Transpose to (H, W, 3) and normalize for display
rgb = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
p98 = np.percentile(rgb, 98)
rgb = np.clip(rgb / p98, 0, 1)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(rgb)
ax.set_title("Sentinel-2 RGB composite (B4, B3, B2) — LU000")
ax.axis("off")
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt

# Transpose to (H, W, 3) and normalize for display
rgb = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(rgb)
ax.set_title("Sentinel-2 RGB composite (B4, B3, B2) — LU000 (no normalization)")
ax.axis("off")
plt.tight_layout()
plt.show()

# %%

with rasterio.open(tile_url) as src:
    print(src.profile)
    false_colour_data = src.read([8, 4, 3])

false_colour = np.transpose(false_colour_data, (1, 2, 0)).astype(np.float32)
p98 = np.percentile(false_colour, 98)
false_colour = np.clip(false_colour / p98, 0, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(rgb)
ax1.set_title("Sentinel-2 RGB (B4, B3, B2)")
ax1.axis("off")
ax2.imshow(false_colour)
ax2.set_title("Sentinel-2 False Colour (B8, B4, B3)")
ax2.axis("off")
plt.tight_layout()
plt.show()

# %%
with rasterio.open(tile_url) as src:
    nir = src.read(8).astype(np.float32)
    red = src.read(4).astype(np.float32)

ndvi = np.where(nir + red == 0, 0, (nir - red) / (nir + red))

fig, ax = plt.subplots(figsize=(5, 5))

im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI")
ax.set_title("Sentinel-2 NDVI")
ax.axis("off")

plt.tight_layout()
plt.show()

# %%
import requests
import geopandas as gpd
from shapely.geometry import Point

# Step 1: Geocode the city name
response = requests.get(
    "https://nominatim.openstreetmap.org/search",
    params={"q": "Kiti, Larnaca, Cyprus", "format": "json", "limit": 1},
    headers={"User-Agent": "funathon-project3"},
)
result = response.json()[0]
lon, lat = float(result["lon"]), float(result["lat"])
print(f"Eurostat coordinates: lon={lon}, lat={lat}")

# Step 2: Create a GeoDataFrame with the point in WGS84, then reproject
city_point = gpd.GeoDataFrame(
    {"city": ["Luxembourg"]}, geometry=[Point(lon, lat)], crs="EPSG:4326"
)
city_point = city_point.to_crs("EPSG:3035")

# Step 3: Load NUTS3 boundaries and spatial join
nuts_url = (
    "https://gisco-services.ec.europa.eu/distribution/v2/"
    "nuts/geojson/NUTS_RG_01M_2021_3035_LEVL_3.geojson"
)
nuts = gpd.read_file(nuts_url)
city_nuts = gpd.sjoin(city_point, nuts, predicate="within")
nuts_code = city_nuts.iloc[0]["NUTS_ID"]
print(f"NUTS3 region: {nuts_code}")  # → LU000

base_url = f"s3://projet-funathon/2026/project3/data/images/{nuts_code}"
print(base_url)

# %%
import pandas as pd
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Build the URL to the parquet index
year = 2024
# nuts_code = "LU000"
parquet_url = (
    f"https://minio.lab.sspcloud.fr/projet-funathon/2026/"
    f"project3/data/images/{nuts_code}/{year}/filename2bbox.parquet"
)

# Read the tile index
tiles = pd.read_parquet(parquet_url)
print(f"{len(tiles)} tiles in {nuts_code}/{year}")

# Get city coordinates in EPSG:3035
x = city_point.geometry.iloc[0].x
y = city_point.geometry.iloc[0].y
print(f"City point (EPSG:3035): x={x:.0f}, y={y:.0f}")

# Find the tile whose bbox contains the city point
tile_filename = None
for _, row in tiles.iterrows():
    xmin, ymin, xmax, ymax = row["bbox"]
    if xmin <= x <= xmax and ymin <= y <= ymax:
        tile_filename = row["filename"]
        break

print(f"Matching tile: {tile_filename}")

# Build the full HTTPS URL
tile_url = (
    f"https://minio.lab.sspcloud.fr/projet-funathon/2026/"
    f"project3/data/images/{nuts_code}/{year}/{tile_filename}"
)

# Open the tile and display the RGB composite
with rasterio.open(tile_url) as src:
    rgb_data = src.read([4, 3, 2])  # Red, Green, Blue bands
    tile_crs = src.crs
    tile_bounds = src.bounds

rgb = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(rgb)
ax.set_title(f"Sentinel-2 — {tile_filename}")
ax.axis("off")
plt.tight_layout()
plt.show()

# %%
import geopandas as gpd
from shapely.geometry import box

tile_gdf = gpd.GeoDataFrame(
    {"tile": ["LU000"], "year": [2024]},
    geometry=[box(*tile_bounds)],
    crs=tile_crs,
)
tile_gdf

# %%
fig, ax = plt.subplots(figsize=(6, 6))
extent = [tile_bounds.left, tile_bounds.right, tile_bounds.bottom, tile_bounds.top]
ax.imshow(rgb, extent=extent)
tile_gdf.boundary.plot(ax=ax, color="red", linewidth=2)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_title("Sentinel-2 tile with boundary overlay (EPSG:3035)")
plt.tight_layout()
plt.show()

# %%
gdf_wgs84 = tile_gdf.to_crs("EPSG:4326")

print("EPSG:3035 bounds:")
print(tile_gdf.total_bounds)
print("\nEPSG:4326 bounds:")
print(gdf_wgs84.total_bounds)

# %%
# nuts_url = (
#     "https://gisco-services.ec.europa.eu/distribution/v2/"
#     "nuts/geojson/NUTS_RG_01M_2021_3035_LEVL_3.geojson"
# )
# nuts = gpd.read_file(nuts_url)

tile_gdf = gpd.GeoDataFrame(
    {"tile": ["LU000"], "year": [2024]},
    geometry=[box(*tile_bounds)],
    crs=tile_crs,
)

joined = gpd.sjoin(tile_gdf, nuts, predicate="intersects")
for _, row in joined.iterrows():
    print(row["NUTS_ID"], row["NUTS_NAME"])

print(joined.area / 1e6)

# %%
from rasterio.warp import transform_bounds

west, south, east, north = transform_bounds(
    tile_crs, "EPSG:4326", *tile_bounds
)

print(f"WGS84 extent: W={west:.4f}, S={south:.4f}, E={east:.4f}, N={north:.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(rgb, extent=[west, east, south, north])
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Sentinel-2 tile in WGS84 coordinates")
plt.tight_layout()
plt.show()

# %%
import folium
from folium.raster_layers import ImageOverlay

center_lat = (south + north) / 2
center_lon = (west + east) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

ImageOverlay(
    image=rgb,
    bounds=[[south, west], [north, east]],
    opacity=0.7,
).add_to(m)

m.save("map.html")
m

# %%
import urllib.request
import io
import numpy as np

# Label URL for a LU000 patch, year 2021
label_url = f"https://minio.lab.sspcloud.fr/projet-funathon/2026/project3/data/labels/{nuts_code}/2021/{tile_filename[:-3]}npy"

with urllib.request.urlopen(label_url) as response:
    label_array = np.load(io.BytesIO(response.read()))

print(f"Label shape: {label_array.shape}")
print(f"Data type:   {label_array.dtype}")
print(f"Classes:     {np.unique(label_array)}")

# %%
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# CLC+ class names and colours
classes = [
    ("Sealed (1)", "#FF0100"),
    ("Woody -- needle leaved trees (2)", "#238B23"),
    ("Woody -- Broadleaved deciduous trees (3)", "#80FF00"),
    ("Woody -- Broadleaved evergreen trees (4)", "#00FF00"),
    ("Low-growing woody plants (bushes, shrubs) (5)", "#804000"),
    ("Permanent herbaceous (6)", "#CCF24E"),
    ("Periodically herbaceous (7)", "#FEFF80"),
    ("Lichens and mosses (8)", "#FF81FF"),
    ("Non- and sparsely-vegetated (9)", "#BFBFBF"),
    ("Water (10)", "#0080FF"),
]
cmap = ListedColormap([color for _, color in classes])

# Load the matching satellite image
image_url = f"https://minio.lab.sspcloud.fr/projet-funathon/2026/project3/data/images/{nuts_code}/2021/{tile_filename}"

with rasterio.open(image_url) as src:
    rgb_data = src.read([4, 3, 2])

rgb = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)

# Side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB")
axes[0].axis("off")

axes[1].imshow(label_array, cmap=cmap, vmin=1, vmax=10)
axes[1].set_title("CLC+ Backbone label")
axes[1].axis("off")

legend_elements = [
    Patch(facecolor=color, edgecolor="black", label=label)
    for label, color in classes
]
fig.legend(
    handles=legend_elements,
    loc="center right",
    bbox_to_anchor=(1.30, 0.5),
    frameon=True,
)
plt.tight_layout()
plt.show()

# %%
import numpy as np
import urllib.request
import io
import rasterio
import folium
from rasterio.warp import transform_bounds
from matplotlib.colors import to_rgba

classes = [
    ("Sealed (1)", "#FF0100"),
    ("Woody -- needle leaved trees (2)", "#238B23"),
    ("Woody -- Broadleaved deciduous trees (3)", "#80FF00"),
    ("Woody -- Broadleaved evergreen trees (4)", "#00FF00"),
    ("Low-growing woody plants (bushes, shrubs) (5)", "#804000"),
    ("Permanent herbaceous (6)", "#CCF24E"),
    ("Periodically herbaceous (7)", "#FEFF80"),
    ("Lichens and mosses (8)", "#FF81FF"),
    ("Non- and sparsely-vegetated (9)", "#BFBFBF"),
    ("Water (10)", "#0080FF"),
]

# Step 1: Load satellite image
# image_url = (
#     "https://minio.lab.sspcloud.fr/projet-funathon/2026/"
#     "project3/data/images/LU000/"
#     "2021/4017000_2974190_0_402.tif"
# )
with rasterio.open(image_url) as src:
    rgb_data = src.read([4, 3, 2])
    bounds_3035 = src.bounds
    crs = src.crs

rgb_overlay = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
rgb_overlay = np.clip(rgb_overlay / np.percentile(rgb_overlay, 98), 0, 1)

# Step 2: Load the matching label
# label_url = (
#     "https://minio.lab.sspcloud.fr/projet-funathon/2026/"
#     "project3/data/labels/LU000/"
#     "2021/4017000_2974190_0_402.npy"
# )
with urllib.request.urlopen(label_url) as response:
    label = np.load(io.BytesIO(response.read()))

# Step 3: Convert label to RGBA
color_lut = np.zeros((11, 4), dtype=np.float32)
color_lut[0] = [0, 0, 0, 0]
for i, (_, hex_color) in enumerate(classes, start=1):
    color_lut[i] = list(to_rgba(hex_color, alpha=0.7))

label_rgba = color_lut[label]

# Step 4: Reproject bounds to WGS84
west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds_3035)

center_lat = (south + north) / 2
center_lon = (west + east) / 2

# Step 5: Create the map
m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

# Step 6: Add overlays
folium.raster_layers.ImageOverlay(
    image=rgb_overlay,
    bounds=[[south, west], [north, east]],
    name="Sentinel-2 RGB",
).add_to(m)

folium.raster_layers.ImageOverlay(
    image=label_rgba,
    bounds=[[south, west], [north, east]],
    name="CLC+ Label",
    opacity=0.8,
).add_to(m)

# Step 7: Layer control
folium.LayerControl().add_to(m)

m.save("map_2.html")
m
# %%
import requests
import numpy as np
import rasterio
from rasterio.io import MemoryFile

year = 2021
bbox = [3649890, 2331750, 3652390, 2334250]
xmin, ymin, xmax, ymax = bbox
resolution = 10
size_x = int((xmax - xmin) / resolution)
size_y = int((ymax - ymin) / resolution)

export_url = (
    f"https://copernicus.discomap.eea.europa.eu/arcgis/rest/services/CLC_plus/"
    f"CLMS_CLCplus_RASTER_{year}_010m_eu/ImageServer/exportImage"
)

params = {
    "f": "image",
    "bbox": f"{xmin},{ymin},{xmax},{ymax}",
    "bboxSR": "3035",
    "imageSR": "3035",
    "size": f"{size_x},{size_y}",
    "format": "tiff",
}

response = requests.get(export_url, params=params)
response.raise_for_status()

with MemoryFile(response.content) as memfile:
    with memfile.open() as src:
        img_array = src.read(1)

img_array[(img_array == 254) | (img_array == 255)] = 0

# %%
import numpy as np
from src.utils import get_file_system

local_path = "3649890_2331750_0_937.npy"
np.save(local_path, img_array)

fs = get_file_system()

s3_path = (
    "projet-funathon/2026/project3/data/labels/LU000/2021/3649890_2331750_0_937.npy"
)
fs.put(local_path, s3_path)

print(fs.ls("projet-funathon/2026/project3/data/labels/LU000/2021"))

# %%
from src.models.model import SegformerB5

model = SegformerB5(
    # 14 matches the layout of the pre-baked GeoTIFFs: the 12 L2A spectral bands
    # (B10 is dropped by atmospheric correction) plus NDVI and NDWI as derived
    # channels. `src/download_region.py` produces the same layout. If you swap in
    # a different dataset (different band count or different derived layers), this
    # number, the normalisation statistics, and `num_channels` in the SegformerConfig
    # must all change together — otherwise the first patch-embedding layer
    # mis-shapes inputs.
    n_bands=14,
    logits=True,             # return raw logits (not probabilities)
    freeze_encoder=False,    # keep encoder trainable
)

# %%
# 1. Parameters per encoder stage.
# Deeper stages have more channels (C₁ < C₂ < C₃ < C₄), so even with fewer transformer
# blocks they end up holding most of the encoder's capacity. This matters during
# fine-tuning: freezing stages 3–4 alone already locks the bulk of the model's weights.
for i, stage in enumerate(model.segformer.encoder.block):
    n = sum(p.numel() for p in stage.parameters())
    print(f"Stage {i+1}: {n:,} parameters")

# 2. Decoder breakdown.
# The four projection layers are tiny linear maps (one per encoder stage); the real
# decoder cost lives in the linear_fuse + classifier. Seeing how light the head is
# explains why training the decoder alone is fast and resistant to overfitting.
for i, proj in enumerate(model.decode_head.linear_c):
    n = sum(p.numel() for p in proj.parameters())
    print(f"Decoder projection {i+1}: {n:,} parameters")

clf_params = sum(p.numel() for p in model.decode_head.classifier.parameters())
dec_total = sum(p.numel() for p in model.decode_head.parameters())
print(f"Classifier: {clf_params:,} / {dec_total:,} decoder params "
      f"({100*clf_params/dec_total:.1f}%)")

# %%
import torch

B, C, H, W = 2, 14, 512, 512   # batch size, channels, height, width
dummy_input = torch.randn(B, C, H, W)
dummy_labels = torch.randint(0, model.config.num_labels, (B, H, W))

print(f"Input  shape: {tuple(dummy_input.shape)}")
print(f"Labels shape: {tuple(dummy_labels.shape)}")

# %%
model.eval()
with torch.no_grad():
    # Without labels → raw logits at H/4 × W/4
    logits = model(dummy_input)
    print(f"Logits shape (no labels):   {tuple(logits.shape)}")
    # Expected: (2, num_classes, 128, 128)  — quarter resolution

    # With labels → logits upsampled to label resolution
    upsampled = model(dummy_input, dummy_labels)
    print(f"Logits shape (with labels): {tuple(upsampled.shape)}")
    # Expected: (2, num_classes, 512, 512)

# %%
outputs = model.segformer(
    dummy_input,
    output_hidden_states=True,
    return_dict=True,
)

for i, hs in enumerate(outputs.hidden_states):
    print(f"Stage {i+1} hidden state: {tuple(hs.shape)}")

# %%
from src.training.metrics import IOU, positive_rate

# Simulate model output and labels
B, num_classes, H, W = 2, 11, 512, 512
dummy_logits = torch.randn(B, num_classes, H, W)
dummy_labels = torch.randint(0, num_classes, (B, H, W))

iou_mean, iou_building = IOU(dummy_logits, dummy_labels, logits=True)
building_rate = positive_rate(dummy_logits, logits=True)

print(f"Mean IoU:      {iou_mean:.4f}")
print(f"Building IoU:  {iou_building:.4f}")
print(f"Building rate: {building_rate:.4f}  (fraction of pixels predicted as building)")

# %%
import os
from dotenv import load_dotenv
from pytorch_lightning.loggers import MLFlowLogger

load_dotenv()

mlf_logger = MLFlowLogger(
    experiment_name="segformer-satellite",
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
    log_model=True,   # upload the best checkpoint as an MLflow artifact
)
# %%
import folium
from folium.raster_layers import ImageOverlay
from rasterio.warp import transform_bounds
import rasterio
import numpy as np

nuts_code = "CY000"
year = 2024
tile_filename = "6462450_1640330_1_4473.tif"

tile_url = f"https://minio.lab.sspcloud.fr/projet-funathon/2026/project3/data/images/{nuts_code}/{year}/{tile_filename}"

with rasterio.open(tile_url) as src:
    rgb_data = src.read([4, 3, 2])  # Red, Green, Blue bands
    tile_crs = src.crs
    tile_bounds = src.bounds

rgb = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)

west, south, east, north = transform_bounds(
    tile_crs, "EPSG:4326", *tile_bounds
)

center_lat = (south + north) / 2
center_lon = (west + east) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

ImageOverlay(
    image=rgb,
    bounds=[[south, west], [north, east]],
    opacity=0.7,
).add_to(m)

m.save("map.html")

# %%
import folium
from folium.raster_layers import ImageOverlay
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.crs import CRS
import numpy as np

nuts_code = "CY000"
year = "2024"
tile_filename = "6462450_1640330_1_4473.tif"
tile_url = f"https://minio.lab.sspcloud.fr/projet-funathon/2026/project3/data/images/{nuts_code}/{year}/{tile_filename}"

# Lecture et reprojection
with rasterio.open(tile_url) as src:
    src_crs = src.crs
    src_transform = src.transform
    src_bounds = src.bounds
    src_width = src.width
    src_height = src.height
    raw_bands = src.read([4, 3, 2]).astype(np.float32)

dst_crs = CRS.from_epsg(4326)
dst_transform, dst_w, dst_h = calculate_default_transform(
    src_crs, dst_crs, src_width, src_height, *src_bounds

)

rgb_wgs84 = np.zeros((3, dst_h, dst_w), dtype=np.float32)
for i in range(3):
    reproject(
        source=raw_bands[i],
        destination=rgb_wgs84[i],
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,

    )

alpha = (rgb_wgs84.max(axis=0) > 0).astype(np.float32)
rgba = np.dstack([np.transpose(rgb_wgs84, (1, 2, 0)), alpha])
west, south, east, north = transform_bounds(src_crs, dst_crs, *src_bounds)
m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=14)
ImageOverlay(
    image=rgba,
    bounds=[[south, west], [north, east]],
    opacity=0.9,
).add_to(m)

m.save("map_fixed.html")

# %%
