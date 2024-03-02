import sys, os
sys.path.insert(0, 'bart-0.6.00/python')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOOLBOX_PATH"]    = 'bart-0.6.00'
sys.path.append('bart-0.6.00/python')

import glob
import numpy as np
import h5py
import sigpy as sp
from bart import bart
print('loadBart')

output_dir = '.'
input_dir = '/data0/chentao/data/fastMRI_knee_test/T1_data'

def main(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = sorted(glob.glob(input_dir + '/*.h5'))

    for file in file_list:
        print('*********Load next MRI Data************')
        # Load specific slice from specific scan
        basename = os.path.basename( file ) 
        output_name = os.path.join( output_dir, basename )
        # if os.path.exists( output_name ):
        #     continue
        with h5py.File(file, 'r') as data:
            #num_slices = int(data.attrs['num_slices']) 368x302x256x18
            kspace = np.array( data['kspace'] ) # 40x15x640x368 
            print('kspace shape:',kspace.shape)
            s_maps = np.zeros( kspace.shape, dtype = kspace.dtype)
            num_slices = kspace.shape[0]
            num_coils = kspace.shape[1]
            for slice_idx in range( num_slices ):
                gt_ksp = kspace[slice_idx]
                s_maps_ind = bart(1, 'ecalib -m1 -W -c0', gt_ksp.transpose((1, 2, 0))[None,...]).transpose( (3, 1, 2, 0)).squeeze()
                s_maps[ slice_idx ] = s_maps_ind

            h5 = h5py.File( output_name, 'w' )
            h5.create_dataset( 's_maps', data = s_maps )
            h5.close()



if __name__ == '__main__':
    main(input_dir,output_dir)
