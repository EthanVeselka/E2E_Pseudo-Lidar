# Processing pipeline

Framework for data processing

(calls = ->)
clean (if needed)
-> process()
-> sample()
-> PLDataset
-> reader 
-> normalizer

returns DataLoader(PLDataset)

## Cleaning

Run clean.py or processing with clean=True to clean data if raw

## Sample Config file 

Configure the sampling parameters in sample_config for data sampling

## Processing

Run processing.py. This function will clean the data if specified, then create a listfile of sample data based on the config paramaters. This Listfile is used to create a PLDataset, which reads in chunks and normalizes data; process() creates DataLoaders from PLDataset and returns them. This function should be called during model train/test initialization.


