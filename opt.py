import torch.optim as optim

scale_embedding = {
    'nr_classes' : 2,
    
    'training_phase' : [
        # {
        #     'nr_epochs' : 30, 
        #     'optimizer'  : [
        #         optim.Adam,
        #         { # should match keyword for parameters within the optimizer
        #             'lr'           : 5.0e-5, # initial learning rate,
        #             'weight_decay' : 0.02
        #         }
        #     ],
        #     'scheduler'  : None, # learning rate scheduler
        #     'train_batch_size' : 4,
        #     'infer_batch_size' : 4,
        #     'freeze' : True,
        #      # path to load, -1 to auto load checkpoint from previous phase, 
        #      # None to start from scratch
        #     'pretrained' : 'resnet50-19c8e357.pth',
        # },
        
        # {
        #     'nr_epochs' : 30, 
        #     'optimizer'  : [
        #         optim.Adam,
        #         { # should match keyword for parameters within the optimizer
        #             'lr'           : 2.5e-5, # initial learning rate,
        #             'weight_decay' : 0.02
        #         }
        #     ],
        #     'scheduler'  : None, # learning rate scheduler
        #     'train_batch_size' : 4,
        #     'infer_batch_size' : 4,
        #     'freeze' : False,
        #      # path to load, -1 to auto load checkpoint from previous phase, 
        #      # None to start from scratch
        #     'pretrained' : -1,
        # },

        {
            'nr_epochs' : 60, 
            'optimizer'  : [
                optim.Adam,
                { # should match keyword for parameters within the optimizer
                    'lr'           : 1.0e-4, # initial learning rate,
                    # 'weight_decay' : 0.02
                }
            ],
            'scheduler'  : lambda x : optim.lr_scheduler.StepLR(x, 30), # learning rate scheduler
            'train_batch_size' : 2,
            'infer_batch_size' : 4,
            'freeze' : False,
             # path to load, -1 to auto load checkpoint from previous phase, 
             # None to start from scratch
            'pretrained' : 'resnet50-19c8e357.pth',
        },

    ],
}

scale_add = {
    'nr_classes' : 2,
    'training_phase' : [{
        'nr_epochs' : 30, 
        'optimizer'  : [
            optim.Adam,
            { # should match keyword for parameters within the optimizer
                'lr'           : 1.0e-4, # initial learning rate,
                'weight_decay' : 0.02 # weight decay is L2 regularizer
            }
        ],
        'scheduler'  : None, # learning rate scheduler
        'train_batch_size' : 4,
        'infer_batch_size' : 4,
        'freeze' : True,
        # path to load, -1 to auto load checkpoint from previous phase, 
        # None to start from scratch
        'pretrained' : 'resnet50-19c8e357.pth',
    }],
}

scale_concat = {
    'nr_classes' : 2,
    'training_phase' : [{
        'nr_epochs' : 30, 
        'optimizer'  : [
            optim.Adam,
            { # should match keyword for parameters within the optimizer
                'lr'           : 1.0e-4, # initial learning rate,
                'weight_decay' : 0.02
            }
        ],
        'scheduler'  : None, # learning rate scheduler
        'train_batch_size' : 4,
        'infer_batch_size' : 4,
        'freeze' : True,
        # path to load, -1 to auto load checkpoint from previous phase, 
        # None to start from scratch
        'pretrained' : 'resnet50-19c8e357.pth',
    }],
}

scale_conv = {
    'nr_classes' : 2,
    'training_phase' : [{
        'nr_epochs' : 30, 
        'optimizer' : [
            optim.Adam,
            { # should match keyword for parameters within the optimizer
                'lr'           : 1.0e-4, # initial learning rate,
                'weight_decay' : 0.02
            }
        ],
        'scheduler'  : None, # learning rate scheduler
        'train_batch_size' : 4,
        'infer_batch_size' : 4,
        'freeze' : True,
        # path to load, -1 to auto load checkpoint from previous phase, 
        # None to start from scratch
        'pretrained' : 'resnet50-19c8e357.pth',
    }],
}

baseline = {
    'nr_classes' : 2,
    'training_phase' : [{
        'nr_epochs' : 30, 
        'optimizer' : [
            optim.Adam,
            { # should match keyword for parameters within the optimizer
                'lr'           : 1.0e-4, # initial learning rate,
                'weight_decay' : 0.02
            }
        ],
        'scheduler'  : None, # learning rate scheduler
        'train_batch_size' : 4,
        'infer_batch_size' : 4,
        'freeze' : True,
        # path to load, -1 to auto load checkpoint from previous phase, 
        # None to start from scratch
        'pretrained' : 'resnet50-19c8e357.pth',
    }],
}
