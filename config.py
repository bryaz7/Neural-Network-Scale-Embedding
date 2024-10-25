
import imgaug # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa

import importlib

#### 
class Config(object):
    def __init__(self):
        self.seed    = 5
        
        # nr of processes for parallel processing input
        self.nr_procs_train = 4 
        self.nr_procs_valid = 2 

        # options: baseline, scale_embedding, scale_add, 
        # scale_concat, scale_conv
        self.exp_mode = 'scale_embedding'

        #### Dynamically setting the config file into variable
        config_file = importlib.import_module('opt')
        config_dict = config_file.__getattribute__(self.exp_mode)
        for variable, value in config_dict.items():
            self.__setattr__(variable, value)

        # migrate these option out
        if self.exp_mode == 'baseline': # aka single scale
            # other case only use scale at x10
            self.data_size  = [2000, 2000]
            self.input_size = [1280, 1280]
        else:
            # index 0 must be of highest magnification level
            # and self.input_size must be of the scale at index 0
            self.scale_list = [20, 15, 10, 7.5, 5]
            self.data_size  = [3000, 3000]
            self.input_size = [1920, 1920]
            if self.exp_mode == 'scale_embedding':
                # assume 10 degree different between two 
                # consecutive magnification level
                self.align_to_scale = 10
                self.scale_diff_angle = 10
                # apply the scale embedding using features extracted 
                # at downsampling level X in resnet (having 5 going 
                # from 1-5). If multiple donwsampling level are provided, 
                # the final features are concatenated and fed to FC classifier
                self.down_sample_level_list = [5, 4]             

        # 
        self.logging = False # False for debug run to test code
        self.log_path = '/mnt/bryan/output/SCALE_EMBEDDING/'
        self.chkpts_prefix = 'model'
        self.model_name = 'v1.0.0.1_%s' % self.exp_mode
        self.log_dir =  self.log_path + self.model_name

    def train_augmentors(self):
        shape_augs = [
            iaa.PadToFixedSize(
                            self.data_size[0], 
                            self.data_size[1], 
                            pad_cval=255,
                            position='center',
                            deterministic=True), 
            iaa.Affine(
                cval=255,
                # scale images to 80-120% of their size, individually per axis
                scale={"x": (0.8, 1.2), 
                       "y": (0.8, 1.2)}, 
                # translate by -A to +A percent (per axis)
                translate_percent={"x": (-0.01, 0.01), 
                                   "y": (-0.01, 0.01)}, 
                rotate=(-179, 179), # rotate by -179 to +179 degrees
                shear=(-5, 5), # shear by -5 to +5 degrees
                order=[0],    # use nearest neighbour
                backend='cv2' # opencv for fast processing
            ),
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 20% of all images
            iaa.CropToFixedSize(self.input_size[0],
                                self.input_size[1],
                                position='center', 
                                deterministic=True)
        ]
        #
        input_augs = [
            iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # gaussian blur with random sigma
                        iaa.MedianBlur(k=(3, 5)), # median with random kernel sizes
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                        ]),
            iaa.Sequential([
                iaa.Add((-26, 26)),
                iaa.AddToHueAndSaturation((-20, 20)),
                iaa.LinearContrast((0.75, 1.25), per_channel=1.0),
            ], random_order=True),
        ]   
        return shape_augs, input_augs

    ####
    def infer_augmentors(self):
        shape_augs = [
            iaa.PadToFixedSize(
                            self.data_size[0], 
                            self.data_size[1], 
                            pad_cval=255,
                            position='center',
                            deterministic=True), 
            iaa.CropToFixedSize(self.input_size[0],
                                self.input_size[1],
                                position='center', 
                                deterministic=True)
            ]
        return shape_augs, None

############################################################################