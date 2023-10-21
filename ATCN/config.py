config = {
    'TNHNK': { # Half output chanel size
        'nhid': [32, 16, 16, 8, 8, 16, 16, 32],
        'sdil': [1,  2,   2,  4,  4,  6,  6, 8],
        'skrn': [32, 16, 16,  8, 8, 4,  4, 2],
        'input_scaling': [1]*8,
        'Paper_name': 'T0'
    },
    'TNH': { # Half output chanel size
        'nhid': [32, 16, 16, 8, 8, 16, 16, 32],
        'sdil': [1,  2,   2,  4,  4,  6,  6, 8],
        'skrn': [64, 32, 32,  16, 16, 8,  8, 4],
        'input_scaling': [1]*8,
        'Paper_name': 'T1'
    },
    'T0': { # Baseline model
        'nhid': [64, 32, 32, 16, 16, 32, 32, 64],
        'sdil': [1,  2,   2,  4,  4,  6,  6, 8],
        'skrn': [64, 32, 32,  16, 16, 8,  8, 4],
        'input_scaling': [1]*8,
        'Paper_name': 'T3'
    },
    'T0NK': { # Same output chanel, half kernel size.
        'nhid': [64, 32, 32, 16, 16, 32, 32, 64],
        'sdil': [1,  2,   2,  4,  4,  6,  6, 8],
        'skrn': [32, 16, 16,  8, 8, 4,  4, 2],
        'input_scaling': [1]*8,
        'Paper_name': 'T2'
    },
    'TN1': {
        'nhid': [128, 64, 64, 32, 32, 64, 64, 128],
        'sdil': [1,  2,   2,  4,  4,  6,  6, 8],
        'skrn': [64, 32, 32,  16, 16, 8,  8, 4],
        'input_scaling': [1]*8,
    },
    'T1': { # This one is training.
        'nhid': [64, 64, 32, 32, 16, 16, 32, 32, 64, 64],
        'sdil': [1, 2,  4,   4,  6,  6,  8,  8, 10, 12],
        'skrn': [64, 64, 32, 32,  16, 16, 8,  8, 4, 2],
        'input_scaling': [1]*10,
    },
    'T2': {
        'nhid': [256, 128, 64, 32, 32, 16, 16, 32, 32, 64, 128, 256],
        'sdil': [1, 2, 4,  6,   6,  8,  8,  10,  10, 12, 14, 16],
        'skrn': [256, 128, 64, 32, 32,  16, 16, 8,  8, 4, 2, 1],
        'input_scaling': [1]*12,
    }

}
