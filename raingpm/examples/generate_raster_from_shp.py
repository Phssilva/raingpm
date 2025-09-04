import os
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import box

def create_raster_from_shapefile(shapefile_path, output_path, pixel_size=5000):
    """
    Cria um raster com pixels de tamanho específico a partir de um shapefile.
    
    Parameters:
    -----------
    shapefile_path : str
        Caminho para o arquivo shapefile
    output_path : str
        Caminho para salvar o raster de saída
    pixel_size : float
        Tamanho do pixel em metros (padrão: 5000 para pixels de 5km)
    """
    # Carrega o shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Garante que o shapefile está em um sistema de coordenadas projetado (metros)
    if gdf.crs and gdf.crs.is_geographic:
        print("Convertendo de coordenadas geográficas para projetadas (UTM)...")
        # Determina a zona UTM apropriada baseada no centroide do shapefile
        centroid = gdf.unary_union.centroid
        utm_zone = int(((centroid.x + 180) / 6) % 60) + 1
        hemisphere = 'south' if centroid.y < 0 else 'north'
        utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
        gdf = gdf.to_crs(utm_crs)
    
    # Obtém os limites do shapefile
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Calcula o número de pixels necessários
    width = int((maxx - minx) / pixel_size) + 1
    height = int((maxy - miny) / pixel_size) + 1
    
    # Cria a transformação para o raster
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Cria um array vazio para o raster
    raster_data = np.zeros((height, width), dtype=np.uint8)
    
    # Rasteriza o shapefile
    shapes = [(geom, 1) for geom in gdf.geometry]
    burned = features.rasterize(
        shapes=shapes,
        out=raster_data,
        transform=transform,
        fill=0,
        all_touched=True
    )
    
    # Cria o perfil do raster
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': rasterio.uint8,
        'crs': gdf.crs,
        'transform': transform,
        'nodata': 0
    }
    
    # Salva o raster
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(burned, 1)
    
    print(f"Raster criado com sucesso em: {output_path}")
    print(f"Resolução: {pixel_size}m x {pixel_size}m")
    print(f"Dimensões: {width} x {height} pixels")
    
    return output_path

# Exemplo de uso:
if __name__ == "__main__":
    shapefile_path = "/home/phsilva/UFSC/raingpm/raingpm/storage/TEMPLATES/roi_poa.shp"
    output_path = "/home/phsilva/UFSC/raingpm/raingpm/storage/TEMPLATES/roi_poa.tif"
    create_raster_from_shapefile(shapefile_path, output_path)