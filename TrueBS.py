from viewshed import Viewshed
from sqlalchemy import create_engine
import csv
import configargparse
import rasterio as rio
from rasterio import mask, features
from rasterio.io import MemoryFile
import numpy as np
from tqdm import tqdm
from shapely.geometry import shape
import shapely.geometry as sg
import shapely.ops as so
import os
import scipy.ndimage as ndimage
import shapely.wkt as wkt
import pickle
from SetCover import set_cover
import geopandas as gpd
import pandas as pd
import networkx as nx


class TrueBS():
    def __init__(self, args):
        self.max_dist = args.max_dist
        self.vs = Viewshed(max_dist=self.max_dist)
        self.DSN = os.environ['DSN']
        self.srid = args.srid  # TODO: check if it is metric
        self.crs = "EPSG:%4d" % (self.srid)
        self.base_dir = args.output
        self.raster_dir = args.raster_dir
        self.poi_elev = args.poi_elev
        self.tgt_elev = args.tgt_elev
        self.ks = args.k
        self.denss = args.dens
        self.comune = args.comune
        self.ratio = args.ratio
        self.dataset_type = args.dataset
        self.sub_area_id = args.sub_area
        self.ranking_type = args.ranking_type
        self.max_build = args.max_build
        self.buildings_table = args.buildings_table
        self.strategy = args.strategy
        self.dump_viewsheds = args.dump_viewsheds

        self.conn = create_engine(self.DSN)
        self.building_mask = rio.open(
            f"{self.raster_dir}/{self.comune.lower()}_buildings_mask.tif", crs=self.crs)
        self.road_mask = rio.open(
            f"{self.raster_dir}/{self.comune.lower()}_roads_mask.tif", crs=self.crs)
        with open(f'{self.comune.lower()}.csv') as sacsv:
            self.subareas_csv = list(csv.reader(sacsv, delimiter=','))
        self.big_dsm = rio.open(
            "%s/%s.tif" % (self.raster_dir, self.comune.lower()), crs=self.crs)
        self.big_dsm_raster = self.big_dsm.read(1)
        self.big_dtm = rio.open(
            "%s/%s_dtm.tif" % (self.raster_dir, self.comune.lower()), crs=self.crs)
        self.big_dtm_raster = self.big_dtm.read(1)

    def get_building(self, id):
        df = self.buildings_df[self.buildings_df.osm_id == id]
        id, obj = list(df.iterrows())[0]
        return obj

    def get_buildings(self, buffer=None):
        if buffer == None:
            buffer = self.buffered_area
        self.buildings_df = gpd.read_postgis(f"SELECT * \
                                            FROM {self.buildings_table}  \
                                            WHERE ST_Intersects('SRID={self.srid};{buffer.wkt}'::geometry, geom) \
                                            ORDER BY gid",
                                             self.conn,
                                             crs=self.crs)
        self.buildings = [b[1] for b in self.buildings_df.iterrows()]
        # self.buildings = sorted(self.BI.get_buildings(shape=self.area, area=self.buffered_area))
        if(len(self.buildings) == 0):
            print("Warning: 0 buildings in the area")
        else:
            print(f"{len(self.buildings)} buildings, {(buffer.area*1e-6):.2f} kmq")

    def convert_coordinates(self, coordinates):
        mapped_coord = self.dataset.index(coordinates[0], coordinates[1])
        coordinates_np = np.array(
            [mapped_coord[0], mapped_coord[1], coordinates[2]], dtype=np.uint32)
        return coordinates_np

    def get_border_points(self, build):
        # remove portion of building that are outiside of area
        cropped_shape = shape(build.geom)  # - self.buffered_area
        # crop mask for selected building
        building, transform = rio.mask.mask(
            self.building_mask, [cropped_shape], crop=True)
        # Binary erosion on building raster to get the border
        mask = building[0] > 0
        struct = ndimage.generate_binary_structure(2, 2)

        # DILATION
        dilate = ndimage.binary_dilation(mask, struct)
        outer_edges = np.logical_and(dilate, np.logical_not(mask))
        rows, cols = np.where(outer_edges)
        coordinates = []
        c = 0
        # Save figure for dilation
        # if build.gid == 747216:
        #     out = np.zeros_like(dilate, dtype=np.uint8)
        #     out = building[0] + dilate
        #     self.save_raster(out, '/home/gabrihacker/747216.tif', transform=transform, nodata=255, nbits=8)
        # get z value from closest building pixel
        for i in range(len(rows)):
            # search in 8 closest pixel:
            x1 = x = rows[i]
            y1 = y = cols[i]
            coords = [(x+1, y), (x-1, y), (x, y+1), (x, y-1),
                      (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]

            # for each one of the 8 adiacent pixel
            for c in coords:
                # if adjacent pixel is in building
                try:
                    if mask[c]:
                        # get height of closest point DSM
                        z_dsm = self.big_dsm_raster[self.big_dsm.index(
                            *rio.transform.xy(transform, *c))]
                        # get height of closest point on DTM
                        z_dtm = self.big_dtm_raster[self.big_dtm.index(
                            *rio.transform.xy(transform, *c))]
                        # If the point is less than 2m ignore it
                        if(z_dsm - z_dtm < 2):
                            continue
                        if(z_dtm + self.poi_elev >= z_dsm):
                            z = z_dsm - 1
                        else:
                            z = z_dtm + self.poi_elev
                        # if height is valid (not out of map)
                        if z > 0:
                        
                            # associate original pixel position to z and pack
                            xx, yy = rio.transform.xy(
                                transform, rows[i], cols[i])
                            coordinates.append((xx, yy, z))
                            if xx < 0 or yy < 0:
                                print("Error")
                            break
                        else:
                            print("Error")
                except IndexError:
                    # Portion of building outside of area
                    continue
        #assert len(coordinates) > 0
        return coordinates

    def get_area(self, sub_area_id):
        for row in self.subareas_csv:
            if int(row[1]) == int(sub_area_id):
                # Read the WKT of this subarea
                self.sub_area = wkt.loads(row[0])
                # Create a buffer of max_d / 2 around it
                self.buffered_area = self.sub_area.buffer(self.max_dist/2)
                # Crop the road mask using the buffer area
                self.road_mask_crop, rm_transform = rio.mask.mask(
                    self.road_mask, [self.buffered_area], crop=True, indexes=1)
                # Generate the transformation matrixes and save them
                self.translation_matrix, self.inv_translation_matrix = self.gen_translation_matrix(
                    self.road_mask_crop)
                os.makedirs(f'{self.base_dir}/{self.comune.lower()}/{self.strategy}/{sub_area_id}', exist_ok=True)
                np.save(f'{self.base_dir}/{self.comune.lower()}/{self.strategy}/{sub_area_id}/translation_matrix', self.translation_matrix)
                np.save(f'{self.base_dir}/{self.comune.lower()}/{self.strategy}/{sub_area_id}/inverse_translation_matrix', self.inv_translation_matrix)
                # Crop and save DSM
                raster, transform1 = rio.mask.mask(
                    self.big_dsm, [self.buffered_area], crop=True, indexes=1)
                with MemoryFile() as memfile:
                    new_dataset = memfile.open(driver='GTiff',
                                               height=raster.shape[0],
                                               width=raster.shape[1],
                                               count=1, dtype=str(raster.dtype),
                                               crs=self.crs,
                                               transform=transform1,
                                               nodata=-9999
                                               )
                    new_dataset.write(raster, 1)
                    new_dataset.close()
                    self.dataset = memfile.open(crs=self.crs)
                    self.dataset_raster = self.dataset.read(1)
                # Crop and save DTM
                raster_dtm, transform2 = rio.mask.mask(
                    self.big_dtm, [self.buffered_area], crop=True, indexes=1)
                with MemoryFile() as memfile:
                    new_dataset = memfile.open(driver='GTiff',
                                               height=raster_dtm.shape[0],
                                               width=raster_dtm.shape[1],
                                               count=1, dtype=str(raster_dtm.dtype),
                                               crs=self.crs,
                                               transform=transform2,
                                               nodata=-9999
                                               )
                    new_dataset.write(raster_dtm, 1)
                    new_dataset.close()
                    self.dataset_dtm = memfile.open(crs=self.crs)
                    self.dataset_dtm_raster = self.dataset_dtm.read(1)
                break

    def gen_translation_matrix(self, road_mask):
        area = int(road_mask.sum() + 1)
        translation_matrix = np.zeros(shape=road_mask.shape, dtype=np.uint32)
        inverse_matrix = np.zeros(shape=(area+1, 2), dtype=np.uint32)
        count = 1
        for i in range(road_mask.shape[0]):
            for j in range(road_mask.shape[1]):
                if road_mask[i, j]:
                    translation_matrix[i, j] = count
                    inverse_matrix[count] = [i, j]
                    count += 1

        return translation_matrix, inverse_matrix

    def optimal_locations(self):
        tot = len(self.sub_area_id) * len(self.ratio) * \
            len(self.denss) * len(self.ks) * len(self.ranking_type)
        i = 0
        for sa_id in self.sub_area_id:
            self.get_area(sa_id)
            self.get_buildings()
            for ratio in self.ratio:
                for dens in self.denss:
                    for k in self.ks:
                        for rt in self.ranking_type:
                            if k == 3 or rt == 'r1':
                                print(
                                    f'{self.comune}_{sa_id} {ratio}% {dens} {rt}_{k} | {i} on {tot}')
                                if self.strategy == 'twostep':
                                    self.twostep_heuristic_tnsm(
                                        sa_id, ratio, dens, k, rt)
                                elif self.strategy == 'threestep':
                                    self.threestep_heuristic_tnsm(
                                        sa_id, ratio, dens, k, rt)
                            i += 1

    def save_results(self, folder, data):
        os.system(f'mkdir -p {folder}')
        self.save_raster(data.astype(
            np.int16), f"{folder}/viewshed_uncropped.tif", transform=self.dataset.transform, nodata=-128)
        save_dataset = rio.open(
            f"{folder}/viewshed_uncropped.tif", 'r', crs=self.crs)
        data, transform1 = rio.mask.mask(
            save_dataset, [self.sub_area], crop=True, indexes=1)
        self.save_raster(data.astype(
            np.int16), f"{folder}/viewshed.tif", transform=transform1, nodata=-128)
        array = data.flatten()
        semipositive = array[array >= 0]
        np.savetxt(f"{folder}/viewshed.csv", semipositive, fmt='%d')

    def twostep_heuristic_tnsm(self, sa_id, ratio, dens, k, ranking_type):
        folder = f'{self.base_dir}/{self.comune.lower()}/{self.strategy}/{sa_id}/{ranking_type}/{k}/{ratio}/{dens}'
        os.system(f'mkdir -p {folder}')

        # Load building ranking
        if ratio < 100:
            with open(f'{self.base_dir}/{self.comune.lower()}/{self.strategy}/{sa_id}/{ranking_type}/{k}/ranking.txt', 'r') as fr:
                selected_buildings = fr.read().split(', ')
                n_buildings = int(len(self.buildings)*ratio/100)
                if n_buildings > len(selected_buildings):
                    print("Don't have so much buildings in ranking")
                else:
                    selected_buildings_ids = selected_buildings[:n_buildings]
                    selected_buildings = [self.get_building(
                        id) for id in selected_buildings_ids]
        else:
            selected_buildings_ids = [b.osm_id for b in self.buildings]
            selected_buildings = [self.get_building(id) for id in selected_buildings_ids]

        # Get coordinates for the choosen buildings
        all_coords = []
        all_buildings = {}
        for build in selected_buildings:
            for c in self.get_border_points(build):
                conv_c = self.convert_coordinates(c)
                if conv_c[0] > 0 and conv_c[1] > 0:
                    all_coords.append(conv_c)
                    all_buildings[(conv_c[0], conv_c[1])] = build.osm_id

        # Check memory availability
        n = len(all_coords)
        total_mem = self.vs.get_memory()  # in bytes
        raster_size = self.road_mask_crop.sum()
        max_viewsheds = int(total_mem / (raster_size))
        if n >= max_viewsheds:
            print(f"Error, too many points in subarea {sa_id}")
            return 0
        # Calculate parallel viewsheds for all points, and save them in compressed array (only points on street)
        cu_mem = self.vs.parallel_viewsheds_translated(self.dataset_raster,
                                                       all_coords,
                                                       self.translation_matrix,
                                                       self.road_mask_crop.sum(),
                                                       0,  # Set poi elev to 0 because we set the height in z
                                                       self.tgt_elev,
                                                       1)
        # Calculate number of BS based on the density and the area size
        n_bs = int(self.buffered_area.area * 1e-6 * dens)
        # Compute set coverage to choose n_bs out of all
        selected_points = set_cover(cu_mem,
                                    n_bs,
                                    k,
                                    ranking_type)

        # Recalculate viewshed only on selected points
        viewsheds = np.zeros(shape=(len(selected_points),
                                    self.dataset_raster.shape[0],
                                    self.dataset_raster.shape[1]),
                             dtype=np.uint8)
        index = []
        for idx, p_i in enumerate(selected_points):
            # Retrieve coords and building of this viewshed and save them
            coord = all_coords[p_i]
            coord_3003 = self.dataset.xy(*coord[:2])
            build = all_buildings[coord[0], coord[1]]
            index.append([coord[0], coord[1], coord[2],
                          coord_3003[0], coord_3003[1], build, p_i])
            # compute viewshed from this point
            viewsheds[idx] = self.vs.single_viewshed(self.dataset_raster,
                                                     coord,
                                                     self.poi_elev,
                                                     self.tgt_elev,
                                                     1).copy_to_host()

        global_viewshed = viewsheds.sum(axis=0)
        used_buildings = set(
            [all_buildings[(all_coords[p_i][0], all_coords[p_i][1])] for p_i in selected_points])

        # Save result
        nodata_result = np.where(
            self.road_mask_crop == 0, -128, global_viewshed)
        self.save_results(folder, nodata_result)
        if self.dump_viewsheds:
            np.save(f'{folder}/viewsheds', viewsheds)
        # Save metrics
        with open(f'{folder}/metrics.csv', 'w') as fw:
            print(
                f"{len(self.buildings)} {len(selected_buildings)} {len(used_buildings)} {len(selected_points)}", file=fw)
        # Save index
        with open(f'{folder}/index.csv', 'w') as fw:
            for r in index:
                print(" ".join(map(str, r)), file=fw)

    def threestep_heuristic_tnsm(self, sa_id, ratio, dens, k, ranking_type):
        folder = f'{self.base_dir}/{self.comune.lower()}/threestep/{sa_id}/{ranking_type}/{k}/{ratio}/{dens}'
        os.system(f'mkdir -p {folder}')

        # Load ranking of buildings
        with open(f'{self.base_dir}/{self.comune.lower()}/threestep/{sa_id}/{ranking_type}/{k}/ranking_{self.max_build}.txt', 'r') as fr:
            selected_buildings = fr.read().split(', ')
            n_buildings = int(len(self.buildings)*ratio/100)
            if n_buildings > len(selected_buildings):
                print(f"Don't have so much buildings in ranking {n_buildings} {len(selected_buildings)}")
            else:
                selected_buildings_ids = selected_buildings[:n_buildings]
                # selected_buildings = [self.get_building(
                #     id) for id in selected_buildings_ids]

        # Load coordinates dict
        with open(f"{self.base_dir}/{self.comune.lower()}/threestep/{sa_id}/coords_ranked.dat", 'rb') as fr:
            coordinates_lists = pickle.load(fr)

        # Pick buildings
        all_coords = []
        all_buildings = {}
        for bid in selected_buildings_ids:
            for conv_c in coordinates_lists[bid]:
                if conv_c[0] > 0 and conv_c[1] > 0:
                    all_coords.append(conv_c)
                    all_buildings[(conv_c[0], conv_c[1])] = bid

        # Check memory availability
        n = len(all_coords)
        # total_mem = self.vs.get_memory()  # in bytes
        # raster_size = self.road_mask_crop.sum()
        # max_viewsheds = int(total_mem / (raster_size))
        # if n >= max_viewsheds:
        #     print(f"Error, too many points in subarea {sa_id}")
        #     return 0
        # else:
        #     print(f"Number of points: {len(all_coords)}")

        # Calculate parallel viewsheds for all points, and save them in compressed array (only points on street)
        cu_mem = self.vs.parallel_viewsheds_translated(self.dataset_raster,
                                                       all_coords,
                                                       self.translation_matrix,
                                                       self.road_mask_crop.sum(),
                                                       0,
                                                       self.tgt_elev,
                                                       1)

        # Calculate visibility graph between all the possible locations
        vg = self.vs.generate_intervisibility_fast(self.dataset_raster, np.array(all_coords))

        # Initialize matrix for coverred points (for maximal set coverage)
        selected_points = []

        # REAL ALGO
        # For each k calcualte coverage and update covered points
        n_bs = int(self.buffered_area.area * 1e-6 * dens)
        selected_points = set_cover(cu_mem,
                                    n_bs,
                                    k,
                                    ranking_type,
                                    vg)

        # Recalculate viewshed only on selected points
        viewsheds = np.zeros(shape=(len(selected_points),
                                    self.dataset_raster.shape[0],
                                    self.dataset_raster.shape[1]),
                             dtype=np.uint8)

        index = []
        for idx, p_i in enumerate(selected_points):
            # Retrieve coords and building of this viewshed and save them
            coord = all_coords[p_i]
            coord_3003 = self.dataset.xy(*coord[:2])
            build = all_buildings[coord[0], coord[1]]
            index.append([coord[0], coord[1], coord[2],
                          coord_3003[0], coord_3003[1], build, p_i])
            # compute viewshed from this point
            viewsheds[idx] = self.vs.single_viewshed(self.dataset_raster,
                                                     coord,
                                                     self.poi_elev,
                                                     self.tgt_elev,
                                                     1).copy_to_host()
            # q

        # Compute global viewshed by summing over the buildings axis
        global_viewshed = viewsheds.sum(axis=0)
        used_buildings = set(
            [all_buildings[(all_coords[p_i][0], all_coords[p_i][1])] for p_i in selected_points])

        # Save result
        nodata_result = np.where(
            self.road_mask_crop == 0, -128, global_viewshed)
        self.save_results(folder, nodata_result)
        if self.dump_viewsheds:
            np.save(f'{folder}/viewsheds', viewsheds)
        # Save metrics
        with open(f'{folder}/metrics.csv', 'w') as fw:
            print(
                f"{len(self.buildings)} {len(selected_buildings)} {len(used_buildings)} {len(selected_points)}", file=fw)
        # Save index
        with open(f'{folder}/index.csv', 'w') as fw:
            for r in index:
                print(" ".join(map(str, r)), file=fw)

    def save_raster(self, data, filename, transform=None, nodata=-9999, nbits=None):
        if not transform:
            transform = self.dataset.transform
        new_dataset = rio.open(filename, 'w', driver='GTiff',
                               height=data.shape[0],
                               width=data.shape[1],
                               count=1, dtype=str(data.dtype),
                               crs=self.crs,
                               transform=transform,
                               nodata=nodata,
                               nbits=nbits
                               )
        new_dataset.write(data, 1)
        new_dataset.close()

    def generate_rankings(self, ks):
        tot = len(ks)*len(self.sub_area_id)*len(self.ranking_type)
        i = 0
        for sa_id in self.sub_area_id:
            for rt in self.ranking_type:
                for k in ks:
                    if k == 3 or rt == 'r1':
                        print(f'{self.comune}_{sa_id}{rt}_{k} | {i} on {tot}')
                        self.get_area(sa_id)
                        self.get_buildings()
                        if self.strategy == 'twostep':
                            self.single_ranking(self.buildings, sa_id, k, rt)
                        elif self.strategy == 'threestep':
                            self.twostep_ranking(self.buildings, sa_id, k, rt)
                    i = i+1

    def single_ranking(self, buildings, sa_id, k, ranking_type):
        folder = f'{self.base_dir}/{self.comune.lower()}/{self.strategy}/{sa_id}'
        os.makedirs(f"{folder}/{ranking_type}/{k}", exist_ok=True)
        # Try to load from cache
        try:
            with open(f"{folder}/coords.dat", 'rb') as fr:
                coordinates_lists = pickle.load(fr)
                print("Coords taken from cache")
        except:
            print("Coords not taken from cache")
            coordinates_lists = []
            for idx, build in enumerate(buildings):
                # get border points and transform from epsg 3003 to local coords of big raster
                coords = [self.convert_coordinates(
                    c) for c in self.get_border_points(build)]
                coordinates_lists.append(coords)
            with open(f"{folder}/coords.dat", 'wb') as fw:
                pickle.dump(coordinates_lists, fw)
        # Calculate cumulative VS for each building (parallely)
        out_mem = self.vs.parallel_cumulative_buildings_vs(self.dataset_raster,
                                                           self.translation_matrix,
                                                           self.road_mask_crop.sum(),
                                                           coordinates_lists,
                                                           0,
                                                           self.tgt_elev,
                                                           1)
        # Find the best k buildings
        selected_buildings = set_cover(out_mem,
                                       len(coordinates_lists),
                                       k,
                                       ranking_type)
        # Take the osm_id and save to disk the ranking
        b_ids = [buildings[i].osm_id for i in selected_buildings]
        with open(f"{folder}/{ranking_type}/{k}/ranking.txt", 'w') as fw:
            fw.write(', '.join(b_ids))

    def twostep_ranking(self, buildings, sa_id, k, ranking_type):
        folder = f'{self.base_dir}/{self.comune.lower()}/{self.strategy}/{sa_id}'
        os.makedirs(f"{folder}/{ranking_type}/{k}", exist_ok=True)
        # Try to load coords from cache
        try:
            with open(f"{folder}/coords_ranked.dat", 'rb') as fr:
                coordinates_dict = pickle.load(fr)
                coordinates_lists = []
                for idx, build in enumerate(buildings):
                    coordinates_lists.append(coordinates_dict[build.osm_id])
                print("Coords taken from cache")
        except FileNotFoundError:
            print("Coords not taken from cache")
            coordinates_dict = {}
            coordinates_lists = []
            for idx, build in enumerate(tqdm(buildings)):
                # get border points and transform from epsg 3003 to local coords of big raster
                coords = [self.convert_coordinates(
                    c) for c in self.get_border_points(build)]
                # save all lists together
                if len(coords) > self.max_build:  # 5 by default
                    out_mem = self.vs.parallel_viewsheds_translated(self.dataset_raster,
                                                                    coords,
                                                                    self.translation_matrix,
                                                                    self.road_mask_crop.sum(),
                                                                    0,
                                                                    self.tgt_elev,
                                                                    1)
                    # For each k calcualte coverage and update covered points
                    selected_points = set_cover(out_mem,
                                                self.max_build,
                                                k,
                                                ranking_type,
                                                tqdm_enable=False)
                    coordinates_dict[build.osm_id] = [coords[i]
                                                      for i in selected_points]
                elif len(coords) > 0:
                    # If there are less than five coords (and more than 0), add all of them
                    coordinates_dict[build.osm_id] = coords
                else:
                    # Otherwise add an empty list to avoid shifting everything
                    coordinates_dict[build.osm_id] = []
                coordinates_lists.append(coordinates_dict[build.osm_id])

            with open(f"{folder}/coords_ranked.dat", 'wb') as fw:
                pickle.dump(coordinates_dict, fw)
        # Calculate the cumulative viewshed among 5 points per building
        out_mem = self.vs.parallel_cumulative_buildings_vs(self.dataset_raster,
                                                           self.translation_matrix,
                                                           self.road_mask_crop.sum(),
                                                           coordinates_lists,
                                                           0,
                                                           self.tgt_elev,
                                                           1)
        # Calculate the buildings' ranking
        selected_buildings = set_cover(out_mem,
                                       len(coordinates_lists),
                                       k,
                                       ranking_type)

        b_ids = [buildings[i].osm_id for i in selected_buildings]
        with open(f"{folder}/{ranking_type}/{k}/ranking_{self.max_build}.txt", 'w') as fw:
            fw.write(', '.join(b_ids))

    def network(self):
        i = 0
        tot = len(self.sub_area_id) * len(self.ratio) * \
            len(self.denss) * len(self.ks) * len(self.ranking_type)
        for sa_id in self.sub_area_id:
            self.get_area(sa_id)
            for ratio in self.ratio:
                for dens in self.denss:
                    for k in self.ks:
                        for rt in self.ranking_type:
                            if k == 3 or rt == 'r1':
                                print(f'{self.comune}_{sa_id} {ratio}% {dens} {rt}_{k} | {i} on {tot}')
                                self.connect_network(sa_id, ratio, dens, k, rt)
                            i += 1

    def calc_visibility_metric_bs(self, folder):
        viewsheds = np.load(f'{folder}/viewsheds.npy')
        print(viewsheds.shape)
        tot = np.sum(viewsheds, axis=0)
        weights = {}
        for i in range(viewsheds.shape[0]):
            weights[i] = {'demand': np.sum(viewsheds[i]), 
                          'weighted_demand': np.nansum(viewsheds[i] / tot)}
        return weights



    def connect_network(self, sa_id, ratio, dens, k, ranking_type):
        # Generate the intervisibility graph
        folder = f'{self.base_dir}/{self.comune.lower()}/{self.strategy}/{sa_id}/{ranking_type}/{k}/{ratio}/{dens}'
        indexes = pd.read_csv(f'{folder}/index.csv', delimiter=' ', header=None, names=['x', 'y', 'z', 'x_3003', 'y_3003', 'building_id', 'p_id'])
        indexes.reset_index(inplace=True) #Keep the id inside of the df
        coordinates = indexes[['x', 'y', 'z']].values
        import pdb; pdb.set_trace()
        vis_mat = self.vs.generate_intervisibility_fast(self.dataset_raster, coordinates).copy_to_host()
        vg = nx.from_numpy_matrix(vis_mat)
        nx.set_node_attributes(vg, indexes.to_dict('index'))
        
        #Add distance
        for src, dst in vg.edges():
            vg[src][dst]['distance'] = np.linalg.norm(coordinates[dst]-coordinates[src])

        #Add metric for number of m2 seen by each bs
        demand_dict = self.calc_visibility_metric_bs(folder)
        nx.set_node_attributes(vg, demand_dict)

        nx.write_graphml(vg, f'{folder}/visibility.graphml.gz')


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(
        description='Truenets utility for offline intervisibility', default_config_files=['sim.yaml'])
    parser.add_argument("-c", "--comune",
                        help="Nome del comune da analizzare",
                        required=True)
    parser.add_argument("-d", "--dataset",
                        help="raster dataset to use (ctr or osm)",
                        default="osm")
    parser.add_argument("-r", "--raster_dir",
                        help="Percorso della cartella contenente i rasters",
                        required=True)
    parser.add_argument("-o", "--output", help="dir for output files",
                        required=True)
    parser.add_argument("-pe", "--poi_elev", help="height of both poles",
                        type=float,
                        required=True)
    parser.add_argument("-te", "--tgt_elev", help="height of both poles",
                        type=float,
                        required=False)
    parser.add_argument("-md", "--max_dist", help="maximum distance for LOS communication",
                        type=float,
                        required=False)
    parser.add_argument("-gid", type=int, required=False, action='append')
    parser.add_argument("--dens", type=int, required=False,
                        action='append', help="BS for sqkm")

    parser.add_argument("--ratio", type=float,
                        required=True, default=[], action='append')

    parser.add_argument("-sa", '--sub_area', required=True,
                        type=str, action='append')
    parser.add_argument("-rt", '--ranking_type',
                        required=True, type=str, action='append')
    parser.add_argument("-k", '--k', required=True, type=int, action='append')
    parser.add_argument("-mb", '--max_build',
                        required=False, type=int, default=5)
    parser.add_argument("--ranking", action='store_true')
    parser.add_argument("--optimal_locations", action='store_true')
    parser.add_argument("--network", action='store_true')
    parser.add_argument("--srid", type=int, default=3003)
    parser.add_argument("--buildings_table", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--dump_viewsheds", action='store_true')

    args = parser.parse_args()


    tn = TrueBS(args)

    if args.ranking:
        tn.generate_rankings(args.k)
    if args.optimal_locations:
        tn.optimal_locations()
    if args.network:
        tn.network()
