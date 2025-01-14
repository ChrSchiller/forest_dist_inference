##### this script reads the list of FORCE tiles containing forest in Germany
##### and loops over them to invoke the Python script 08_sits_bert_prediction_on_force_tiles.py
##### to predict the disturbance estimates for each tile

import os
import subprocess
import argparse

if __name__ == "__main__":

    ### create the parser
    parser = argparse.ArgumentParser()

    ### path to inference script for single tile
    parser.add_argument('--inference_script_path', type=str, required=True)

    ### path to the model (trained on LUX study site as spatial hold-out, meaning that it is most representative for Germany)
    parser.add_argument('--model_path', type=str, required=True)

    ### force tile base path
    ### default directory is the default directory the FORCE datacube would have on eo-lab.org
    parser.add_argument('--force_tile_base_path', type=str, required=False, default='/force/FORCE/C1/L2/ard')

    ### result tile base path to store the tile directories and the results
    parser.add_argument('--result_tile_base_path', type=str, required=True)

    ### forest mask base path containing the tile folders and the forest mask.tif
    ### forest mask was created by R script create_forest_mask_by_force_tiles.R
    parser.add_argument('--forest_mask_base_path', type=str, required=True)

    ### end date of time series
    ### as a string in the format YYYYMMDD
    parser.add_argument('--end_date', type=str, required=False)

    ### parse the arguments
    args = parser.parse_args()

    ### get the tiles
    tiles = os.listdir(args.forest_mask_base_path)
    tiles = [tiles[tile] for tile in range(0, len(tiles))]
    # ### if two VM's are used, just split into two parts: 
    # tiles = tiles[238:len(tiles)] # second half of FORCE tiles

    print('There are ' + str(len(tiles)) + ' FORCE tiles containing forest in Germany!')
    print(len(tiles))

    ### loop over tiles
    for tile in tiles:
        print('Next tile to process: ' + tile)

        ### invoke the Python script using the defined arguments
        ### if args.end_date is not empty: 
        if args.end_date:
            command = ['python3', args.inference_script_path, 
                   '--tile_name', tile, 
                   '--model_path', args.model_path, 
                   '--force_tile_base_path', args.force_tile_base_path, 
                   '--result_tile_base_path', args.result_tile_base_path, 
                   '--forest_mask_base_path', args.forest_mask_base_path, 
                   '--end_date', args.end_date]
        else:
            command = ['python3', args.inference_script_path, 
                   '--tile_name', tile, 
                   '--model_path', args.model_path, 
                   '--force_tile_base_path', args.force_tile_base_path, 
                   '--result_tile_base_path', args.result_tile_base_path, 
                   '--forest_mask_base_path', args.forest_mask_base_path]
        ### execute the subprocess
        subprocess.call(command)

        print('Finished processing tile ' + tile)
