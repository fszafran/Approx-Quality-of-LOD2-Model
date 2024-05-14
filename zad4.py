import laspy
from bs4 import BeautifulSoup
import pandas as pd
# import numpy as np

if __name__ == '__main__':
    # odczyt pliku laz - chmura punktów
    # las_file = laspy.read('lidar.laz')
    # bpoints = las_file.points[las_file.classification == 6]

    # zapis współrzędnych punktów do osobnego pliku
    # with open('building_point_cloud.txt', 'w') as f:
    #     f.write(f"X Y Z\n")
    #     for point in bpoints:
    #         f.write(f"{point.X} {point.Y} {point.Z}\n")

    # odczyt pliku gml
    with open('lod_2_jeden_budynek.gml', 'r') as f:
        xml_file = f.read()
    soup = BeautifulSoup(xml_file, 'xml')

    # wyekstraktowanie współrzędnych punktów z gml/xml-a
    roofs = soup.find_all('RoofSurface')
    roof_data = []
    polygon_id = 0
    for roof in roofs:
        poly_text = roof.find('Polygon').text if roof.find('Polygon') else None
        poly_lines = poly_text.strip().split('\n')
        # print(poly_text)
        for line in poly_lines:
            if not line.strip():
                continue
            x, y, z = map(float, line.split())
            roof_data.append({'ID': polygon_id, 'x': x, 'y': y, 'z': z})
        polygon_id += 1

    # zapis punktów gml do osobnego pliku w formacie: id_poligonu - x - y - z
    roof_df = pd.DataFrame(roof_data)
    # roof_df.to_csv('roof_points.txt', index=False, sep=' ')

    max_x, min_x, max_y, min_y = roof_df['x'].max(), roof_df['x'].min(), roof_df['y'].max(), roof_df['y'].min()

    # with open('building_point_cloud.txt', 'r') as infile:
    #     next(infile)
    #     with open('filtered_building_point_cloud.txt', 'w') as outfile:
    #         for line in infile:
    #             x, y, z = map(float, line.split())
    #             if min_x <= x <= max_x and min_y <= y <= max_y:
    #                 outfile.write(f"{x} {y} {z}\n")



