### this script is used to predict forest disturbance on a single FORCE tile
### it is used in the inference_on_all_force_tiles.py script to loop over all FORCE tiles
### and predict the forest disturbance for each tile

import torch
from torch.utils.data import DataLoader

from model.bert import SBERT
from predictor.predictor import SBERTPredictor
from dataset.predict_dataset import PredictDataset
from model.classification_model import SBERTClassification

import rasterio
from pandas.tseries.offsets import DateOffset
from datetime import datetime
import numpy as np
import argparse
import os
import time
import csv
import pandas as pd
import copy
from utils.read_mask_force_rasters import read_mask_force_rasters


if __name__ == "__main__":


    ### create the parser
    parser = argparse.ArgumentParser()

    ### tile name
    parser.add_argument('--tile_name', type=str, required=True)

    ### path to the model (trained on LUX study site as spatial hold-out, meaning that it is most representative for Germany)
    parser.add_argument('--model_path', type=str, required=True)
    
    ### path to FORCE tile directories
    parser.add_argument('--force_tile_base_path', type=str, required=False, default='/force/FORCE/C1/L2/ard')

    ### where to save the results?
    parser.add_argument('--result_tile_base_path', type=str, required=True)

    ### forest mask base path containing the tile folders and the forest mask.tif
    ### this will ensure that each tile does contain forest (as it is pre-filtered)
    parser.add_argument('--forest_mask_base_path', type=str, required=True)

    ## end date of time series
    parser.add_argument('--end_date', type=str, required=False)

    ### parse the argument
    args = parser.parse_args()

    start_time = time.perf_counter()

    # print('CUDA available: ')
    # print(torch.cuda.is_available())

    print("Current time (beginning of script):")
    print(datetime.now())

    ### model parameters from training
    ### they are hard-coded here because this is how the model has been defined
    BATCH_SIZE = 128
    HIDDEN_SIZE = 128  # also called (and equals) embedding dim in the code
    N_LAYERS = 3  
    ATTN_HEADS = 8  # note that HIDDEN_SIZE % ATTN_HEADS must equal 0
    DROPOUT = 0.3  # not used in inference, but must be defined
    NUM_CLASSES = 1
    global SEQ_LEN
    SEQ_LEN = 256
    NUM_WORKERS = 12 # 12 == max on eolab platform
    FEATURE_NUM = 10  # 10 bands

    ### additional variables that need to be defined
    ### forest mask was created by R script create_forest_mask_by_force_tiles.R
    ### using Copernicus Land Monitoring Service forest mask and the FORCE datacube on eolab.org
    FOREST_MASK = os.path.join(args.forest_mask_base_path, args.tile_name, 'forest_mask.tif')
    RESULT_PATH = os.path.join(args.result_tile_base_path, args.tile_name)

    ### create RESULT_PATH
    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    if not os.path.exists(os.path.join(args.result_tile_base_path, '..', 'last_obs_valid')):
        os.mkdir(os.path.join(args.result_tile_base_path, '..', 'last_obs_valid'))
    if not os.path.exists(os.path.join(args.result_tile_base_path, '..', 'last_obs_valid', args.tile_name)):
        os.mkdir(os.path.join(args.result_tile_base_path, '..', 'last_obs_valid', args.tile_name))

    ### filter force tile for: S2A/B and relevant timesteps
    files = os.listdir(os.path.join(args.force_tile_base_path, args.tile_name))
    boa_filenames = list(sorted(filter(lambda x: 'SEN2' in x and 'BOA' in x, files)))

    ### define end date either by user or by last observation in the FORCE tile
    if args.end_date:
        ### end date provided by user
        ENDDATE = args.end_date
    else: 
        ### last date of time series/scenes in FORCE tile directory
        ENDDATE = boa_filenames[len(boa_filenames)-1][:8]
    ### start date defined as 4 years before end date
    STARTDATE = pd.to_datetime(pd.to_datetime(ENDDATE, format='%Y%m%d') - DateOffset(days=1460))  # 4 years
    print(STARTDATE)
    print(ENDDATE)

    ### get dates
    global dates
    dates = [pd.to_datetime(s[:8], format='%Y%m%d') for s in boa_filenames]

    ### check which dates are later than startdate (others can be discarded)
    dates = [t for t in dates if t >= STARTDATE]

    ### use this information to filter the list of S2 scenes
    ### discard files before start date
    ### we want as few rasters to load as possible
    boa_filenames = boa_filenames[-len(dates):]

    ### also discard all observations later than enddate
    dates = [t for t in dates if t <= pd.to_datetime(pd.to_datetime(ENDDATE, format='%Y%m%d'))]
    boa_filenames = boa_filenames[:len(dates)]

    ### read forest mask
    with rasterio.open(FOREST_MASK) as frst:
        forest_mask = np.squeeze(frst.read())

    ### stack all relevant rasters
    ### filter with forest mask while stacking
    ### use numpy arrays instead of rasters right away
    print('Stacking all FORCE tiles in study period...')
    print("Current time:")
    print(datetime.now())

    ### create numpy array to store all rasters
    h, w = forest_mask.shape
    all_boas = np.empty((h, w, len(boa_filenames), FEATURE_NUM), dtype=np.float32)

    ### read all the files and write to raster stack (= numpy array)
    for (i, f) in enumerate(boa_filenames):
        all_boas[:, :, i, :] = read_mask_force_rasters(os.path.join(args.force_tile_base_path, args.tile_name, f), forest_mask)

    ### get quality information: is last observation valid, or masked?
    ### will be interesting for the users and early warning (and more convenient than finding out themselves):
    ### if there was no new observation, prediction will be the same -> indicate in new "quality" layer
    last_obs_valid = copy.deepcopy(all_boas[:, :, len(boa_filenames)-1, 0]) # get values of last scene, only one channel
    ### distill information by:
    ### - non-forest = np.nan (should be the case already)
    ### - forest + masked (e.g. by cloud) = 0
    ### - forest + not masked (no clouds, etc.) = 1
    mask = (forest_mask == 1) & np.isnan(last_obs_valid)
    last_obs_valid[mask] = 0 # forest pixel + masked observation (clouds, etc.)
    mask = (forest_mask == 1) & ~np.isnan(last_obs_valid)
    last_obs_valid[mask] = 1 # forest pixel + valid observation (no clouds, etc.)

    ### read the raster data into a NumPy array
    with rasterio.open(FOREST_MASK) as lov:
        raster_data = lov.read(1)

    ### write the numpy array data to the raster file
    with rasterio.open(os.path.join(args.result_tile_base_path, '..', 'last_obs_valid', args.tile_name, 'last_obs_valid_' + ENDDATE + '.tif'), 'w', **lov.meta) as lov_dst:
        lov_dst.write(last_obs_valid, 1)

    ### reshape by flattening over height and width
    ### makes it easier to loop through dataset parallely
    all_boas = all_boas.reshape((-1, len(boa_filenames), FEATURE_NUM))

    ### load and initialize model
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sbert = SBERT(num_features=FEATURE_NUM, hidden=HIDDEN_SIZE, n_layers=N_LAYERS,
                  attn_heads=ATTN_HEADS, dropout=DROPOUT)
    global model
    model = SBERTClassification(sbert, NUM_CLASSES, SEQ_LEN)  # .to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    ### locations of all pixels that should be considered for prediction are exactly the non-NA
    ### values in forest_mask!
    forest_mask = forest_mask.reshape((-1)) # flatten the forest mask
    iter_locations = np.where(forest_mask == 1)[0]

    ### initialize Dataset class, DataLoader and model instance
    predict_dataset = PredictDataset(iter_locations, all_boas, dates, SEQ_LEN, FEATURE_NUM)
    print("Creating Dataloader...")
    predict_data_loader = DataLoader(predict_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,
                    shuffle=False, drop_last=False)
    print("prediction samples: %d" %
          (len(predict_dataset)))
    trainer = SBERTPredictor(sbert, NUM_CLASSES, seq_len=SEQ_LEN,
                             train_dataloader=predict_data_loader,
                             valid_dataloader=predict_data_loader)
    ### obtain the saved model from trainer.load
    trainer.load(args.model_path)

    print('Predicting...')
    print("Current time:")
    print(datetime.now())

    ### only returning raw predictions (not classes to save on memory)
    pred_raw = trainer.predict(predict_data_loader)

    print('Predicting finished!')
    print("Current time:")
    print(datetime.now())
    ### there might be the case that the last batch has only one value, resulting in ndim == 0
    ### then pred_raw[-1].shape == 0
    ### in this case, we have to add a dimension to the array
    if pred_raw[-1].ndim == 0:
        pred_raw[-1] = np.expand_dims(pred_raw[-1], axis = 0)
    pred_raw = np.concatenate(pred_raw, axis=0)

    ### we can use the forest_mask array, because it is not needed anymore, 
    ### still uses memory and has the correct shape
    forest_mask[iter_locations] = pred_raw

    ### reshape to correct (original) format
    result_np = forest_mask.reshape((h, w))

    ### at this point, it would be possible to clear the memory from most of the data
    ### (if RAM is an issue)

    ### read the raster data into a NumPy array
    with rasterio.open(FOREST_MASK) as src:
        raster_data = src.read(1)

    # write the numpy array data to the raster file
    with rasterio.open(os.path.join(RESULT_PATH, 'preds_raw_' + ENDDATE + '.tif'), 'w', **src.meta) as dst:
        dst.write(result_np, 1)

    ### create meta directory for elapsed time estimates
    if not os.path.exists(os.path.join(args.result_tile_base_path, '../meta')):
        os.mkdir(os.path.join(args.result_tile_base_path, '../meta'))

    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) / 60

    # open the csv file in append mode
    with open(os.path.join(args.result_tile_base_path, '../meta/elapsed_times.csv'), 'a') as csvfile:
        writer = csv.writer(csvfile)
        # write the ID and elapsed time to the CSV file
        writer.writerow([args.tile_name, elapsed_time])

    print('FORCE tile ' + args.tile_name + ' finished and written to disk! Exiting...')
    print("Current time:")
    print(datetime.now())