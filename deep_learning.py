import numpy as np
import sys
sys.path.append('E:\metametametabci\metabci')
from brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from brainda.algorithms.deep_learning.tsception import TSception
from brainda.paradigms import Video
from brainda.datasets.MuseData import MuseData
dataset = MuseData()  # declare the dataset
paradigm = Video(
    channels=['Fp7', 'Fp8'],
    events=None,
    intervals=None,
    srate=None
)  # declare the paradigm, use recommended Options

# X,y are numpy array and meta is pandas dataFrame
X, y, meta = paradigm.get_data(
    dataset,
    subject_id=1,
    return_concat=True,
    n_jobs=None,
    verbose=False)

set_random_seeds(38)
kfold = 5
indices = generate_kfold_indices(meta, kfold=kfold)

# assume we have a X with size [batch size, number of channels, number of sample points]
# for shallownet/deepnet/eegnet, you can write like this: estimator = EEGNet(X.shape[1], X.shape[2], 2)

# for GuneyNet, you will have a X with size
# [batch size, number of channels, number of sample points, number of sub_bands]
# and you need to transpose it or in other words switch the dimension of X
# to make X size be like [batch size, number of sub_bands, number of channels, number of sample points]
# and initialize guney like this: estimator = GuneyNet(X.shape[2], X.shape[3], 2, 3)

# for convCA, you will also need a T(reference signal), you can initialize network like shallownet by
# estimator = ConvCA(X.shape[1], X.shape[2], 2),
# but you need to wrap X and T in a dict like this {'X': X, 'T', T} to train the network
# like this:
# dict = {'X': train_X, 'T', T}
# estimator.fit(dict, train_y).
#
# the size of X and T need to be
# X: [batch size, number of channels, number of sample points]
# T: [batch size, number of channels, number of classes, number of sample points]
estimator = TSception(num_classes=2, input_size=(1,2,256), sampling_rate=256, num_T=15, num_S=15,
                hidden=32, dropout_rate=0.5)

accs = []
for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    p_labels = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
    accs.append(np.mean(p_labels==y[test_ind]))
print(np.mean(accs))

