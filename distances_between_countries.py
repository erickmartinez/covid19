# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 06:28:36 2020

@author: Erick
"""

import geopy.geocoders as geocoders
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import time

gc = geocoders.OpenMapQuest(api_key='K3jJ7vzSucxf3fbSH7QlUrRVrMdOfYqR')
csv_file = './countries_list.csv'
reference_city = 'Wuhan'

if __name__ == "__main__":
    
    df_region_list = pd.read_csv(csv_file)
    df_regions = df_region_list['Country/Region']
    regions = list(df_regions)
    n_cities = len(df_region_list)
    coordinates = np.empty(n_cities, dtype=np.dtype([('latitude', 'd'), 
                               ('longitude', 'd')]))
    distance_from_wuhan = np.empty(n_cities)
    loc_wuhan = gc.geocode('Wuhan')
    coords_wuhan = (loc_wuhan.latitude, loc_wuhan.longitude)
    # get the coordinates
    for i, row in df_region_list.iterrows():
        geoloc = gc.geocode(row['City'])
        print('Got location for \'{0}\': {1}'.format(row['City'], geoloc))
        coordinates[i] = (geoloc.latitude, geoloc.longitude)
        distance_from_wuhan[i] = geodesic(coordinates[i], coords_wuhan).kilometers
        time.sleep(0.001)
    
    # Make a matrix of distances
    distance_matrix = np.empty((n_cities, n_cities))
    for i in range(n_cities):
        c1 = (coordinates[i]['latitude'], coordinates[i]['longitude'])
        for j in range(n_cities):
            c2 = (coordinates[j]['latitude'], coordinates[j]['longitude'])
            distance_matrix[i,j] = geodesic(c1, c2).kilometers
    
    df_distances = pd.DataFrame(data=distance_matrix, columns=list(df_regions))
    
    col_order = regions
    col_order.insert(0,'Country/Region')
    
        
    df_distances['Country/Region'] = df_regions
    df_distances = df_distances[col_order]
    
    
    df_distances.to_csv('./countries_distance_matrix.csv', index=False)
    df_coordinates = pd.DataFrame(data=coordinates)
    df_distance_from_wuhan = pd.DataFrame(data=distance_from_wuhan, columns=['Distance from Wuhan (km)'])
    
    df_region_list['latitude'] = df_coordinates['latitude']
    df_region_list['longitude'] = df_coordinates['longitude']
    df_region_list['distance from Wuhan (km)'] = df_distance_from_wuhan
    df_region_list.to_csv('./countries_coordinates.csv', index=False)
    
    
    
        
    
    