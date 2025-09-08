import h5py
import pandas as pd
import numpy as np
import math

def get_data_filename(deviceDirs = ["oneplus5t", "xiaomi11lite5g"][1],
                      dataPath="../data",
                      mapDirs=["ict_fl1"], 
                      dates = ["24dec24"],
                      file_format = '.h5'
                     ):
    return f"{dataPath}/{mapDirs}/{deviceDirs}/{dates}/combined_coord_raw_mag_{dates}.h5"

def hdf5_read( filename:str ) -> dict:
    d = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            print(key)

            ds_arr = f[key][()]   # returns as a numpy array
            d[key] = ds_arr.flatten() # appends the array in the dict under the key
    return d

def get_sample_size_to_stratify_dataset( df, ref_points ):
    r = []
    for i,(l) in enumerate(ref_points):
        r += [{'Ref#':f'{i}', 'Records':len(df[df['label']==l]), 'Label':f'{l}'}]
    #pd.DataFrame.from_records(r)
    min_set_size = pd.DataFrame.from_records(r)['Records'].min()
    rand_sample_size = .85
    to_create_round_number = 94
    sample_size = int( min_set_size * rand_sample_size ) - to_create_round_number
    return sample_size

def compile_new_batched_df( df, ref_points, 
                           sample_size,
                           batch_size=50,
                           ):
    dfrs = []
    batch_ids = []
    for i, a in enumerate(range(0, sample_size, batch_size)) :
        batch_ids += [i]*batch_size
    for i,(l) in enumerate(ref_points):
        dft = df[df['label']==l].iloc[0:sample_size].copy() # Stratify dataset.
        # dft = df[df['label']==l].sample(n=sample_size).copy() # Stratify dataset.
        batches = np.array_split(dft, len(dft) // batch_size)
        # Assign the batch IDs to dft
        dft['batch_id'] = batch_ids               
        dfrs += [dft]
    dfr = pd.concat( dfrs )
    return dfr

# def compile_new_batched_df( df, ref_points, 
#                            sample_size,
#                            funcs=[np.median],
#                            kwargs_array=None,
#                            features=['X','Y','Z'], 
#                            targets=['Cx','Cy'], 
#                            batch_size=100,
#                            ):
#     dfrs = []
#     for i,(l) in enumerate(ref_points):
#         dft = df[df['label']==l].sample(n=sample_size) # Stratify dataset.
#         dfr = parse_reference_location_samples( dft.copy(), funcs, kwargs_array, batch_size, features, targets )
#         # display(dfr)
#         print(f'#{i}\t', f'\tLabel: {l}', f'Preprocess-size: {len(dft)}', f'Postprocess-size: {len(dfr)}')
#         dfrs += [dfr]
#     dfr = pd.concat( dfrs )
#     return dfr

def parse_reference_location_samples( dft, funcs, kwargs_array, batch_size, features, targets ) -> pd.DataFrame:
    batches = np.array_split(dft, len(dft) // batch_size)
    r = []
    d_target = {k:v for k,v in zip(targets, batches[0].iloc[0][targets]) }
    for b in batches:
        tmp = {}
        
        for func, args in zip(funcs, kwargs_array):
            f_str = func.__name__
            b2 = None
            if args is None:
                b2 = func(b[features])
            else:
                for k,v in args.items():
                    f_str += f'_{k}{v}'
                b2 = func(b[features], **args)
            # print(b[targets])
            # print(b2, len(batches))
            tmp = tmp | {f'{k}_{f_str}':v for k,v in zip(features,b2)}
            
        
        r += [ tmp | d_target ]
        
    return pd.DataFrame.from_records(r)

def get_output_data_filename( fn, replace_str='raw', suffix='batched', ext='xlsx' ):
    ofn = fn.replace(replace_str, suffix )
    return '.'.join(ofn.split('.')[:-1])+f'.{ext}'

def rmsValue(arr):
    square = 0
    mean = 0.0
    root = 0.0
    
    n = len(arr)
     
    #Calculate square
    for i in range(0,n):
        square += (arr[i]**2)
     
    #Calculate Mean 
    mean = (square / (float)(n))
     
    #Calculate Root
    root = math.sqrt(mean)
     
    return root