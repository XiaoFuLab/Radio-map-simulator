# Radio Map Simulator


## Installation:
The code was built with the `python3.9`.

To run the code follow the following installation instructions:
- First install all the required package using following command.

```
pip install -r requirements.txt
```

    

## Usage:
Sample simulation can be run using the following command in linux.
```bash
bash run.sh
```
or 

## Individual parameter using command:
To generate radio map with desired parameters run the following command with your parameter setting:
```bash
python generate_radio_map.py \
  --emitters 6 \
  --shadow-variance 4 \
  --decorrelation-distance 100 \
  --snr 0 \
  --space-x 100 \
  --space-y 100 \
  --bandwidth-length 64 \
  --psd-basis 'g' \
  --save-file 'test.npz' \
  --visualize 'True' 
```

## Parameter description:
Parameter  | Description
------------- | -------------
--emitters  | Number of emitters in radio map
--shadow-variance  | Shadow variance to control shadowing
--decorrelation-distance | Decorrelation distance to control shadowing
--snr | Signal-to-noise ratio if you want to add noise to radio map
--space-x | Map dimension along x-axis 
--space-y | Map dimension along y-axis
--bandwidth-length | Map dimension along frequency axis (Number of frequency bins)
--psd-basis | Power Spectrum Density basis for shape ('g'-> Gaussian, 's'-> Sinc)
--save-file | File name to save the generated map
--visualize | Visualize the radio map (True if you want to plot)