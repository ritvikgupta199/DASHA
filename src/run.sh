#!/bin/bash

DS=$1 
# Genomics: 'enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac', 'promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors','splice_sites_all'
# Satellite Imaging: 'big_earth_net', 'brick_kiln', 'eurosat', 'so2sat', 'forestnet', 'pv4ger', 'BigEarth', 'canadian_cropland', 'fmow'
# Time Series Forecasting: [dataset]_[horizon_length]; 
#       -- datasets: ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'ILI', 'Traffic', 'Weather']
#       -- horizon_length:  [24, 36, 48, 60] for ILI
#                           [96, 192, 336, 720] for others 

cwd=$(pwd)
python main_dasha.py --save_dir "${cwd}/results/" --dataset $DS