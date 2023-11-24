# -*- coding: utf-8 -*-
"""
LAB TEMPLATE

Audio features

"""

# Generic imports
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
# Import ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
# Import audio feature library
import librosa as lbs
# Import our custom lib for audio feature extraction (makes use of librosa)
import audio_features as af

# extra imports 
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
##############################################################################
# Feature extraction function
##############################################################################

def extract_features(X,verbose = True):
    """
    >> Function to be completed by the student
    Extracts a feature matrix for the input data X
    ARGUMENTS:
        X: Audio data file paths to analyze
    """
    # Number of samples to process
    num_data = len(X)
    # Sample rate of the signals    
    sr = lbs.get_samplerate(X[0])
    
    # Specify the number of features to extract
    n_feat = 5
    
    # Generate empty feature matrix
    M = np.zeros((num_data,n_feat))
    
    # Feature extraction parameters
    # Frame length
    flen = 512
    # Number of subframes
    nsub = 10
    # Hop length
    hop = 128
    # Threshold below reference to consider as silence (dB) 
    thr_db = 20
    
    for i in range(num_data):
        if verbose:
            print('%d/%d... ' % (i+1, num_data), end='')
        # Read audio signal
        audio_data,_ = lbs.load(X[i], sr=sr)
        # Preprocessing (Trim + center)
        audio_data = af.preprocess_audio(audio_data, thr=thr_db)
        

        
        # Get first two features (Energy entropy mean and max)
        energy_entropies = af.get_energy_entropy(audio_data, 
                                                     flen=flen, 
                                                     hop=hop, 
                                                     nsub=nsub)
        # Compute mean (ignore nan values)
        M[i,0] = np.nanmean(energy_entropies)
        # Compute max value (ignore nan values)
        M[i,1] = np.nanmax(energy_entropies)
        
        ##########################################
        # Extract additional features
        fl = 2048
        ctsft = lbs.feature.chroma_stft(y=audio_data, hop_length=fl)
        M[i,2] = np.nanmax(ctsft)

        spc = af.get_spectral_centroid(audio_data, .1)
        M[i,3] = np.max(spc) # center of mass of frequency
        # zero_crossings = lbs.zero_crossings(audio_data, pad=False)
        # zero_crossings_rate = np.mean(zero_crossings)
        M[i,4] = np.nanmax(af.get_spectral_entropy(audio_data))
        """
        KEEPING THE SVM
            + WITH THE NANMEAN AND NANMAX ENERGY ENTROPIES, STFT FOR 2048, MEAN SPECTRAL CENTROID 
            AND NANMAX SPECTRAL ENTROPY YOU OBTAIN .58 AUC
            +  adding mfccs goes to .5537
            + nanmean of energy entropies, nanmax of energy entropies, stft 2048 nanmax and max(mean gives worse results .51) of spectral centroid
            gives an AUC of .64

        WITH XGBOOST
            + Best combination so far is 5 features, nanmean and max for energy entropy, 
            max chroma stft 2048, max spc and nanmax spectral entropy, gives .824
        """
        """

        # second experiment
        frame_length_512 = 512
        frame_length_2048 = 2048

        # Extract Chroma STFT features with frame length 512
        # chroma_stft_512 = lbs.feature.chroma_stft(y=audio_data, hop_length=frame_length_512)

        # Extract Chroma STFT features with frame length 2048
        chroma_stft_2048 = lbs.feature.chroma_stft(y=audio_data, hop_length=frame_length_2048)
        M[i,0] = np.nanmax(chroma_stft_2048)
        
        #
        # >>>>> ADD CODE HERE
        #
        #
        ##########################################
        """
        if verbose:
            print('Done')
    return M

##############################################################################
# Data read (and prepare)
##############################################################################

# Get file names
major_files = listdir('./data/Major')
minor_files = listdir('./data/Minor')
major_files = ['./data/Major/' + f for f in major_files]
minor_files = ['./data/Minor/' + f for f in minor_files]

# Unify data and code labels
X = deepcopy(minor_files)
X.extend(major_files)
# Label 0 --> Minor
# Label 1 --> Major (Arbitrary positive class)
y = list(np.concatenate((np.zeros(len(minor_files)), 
                         np.ones(len(major_files))), axis = 0).astype(int))

# Fix size of test set
test_size = 0.3
# Manual seed (for replicability)
ran_seed = 999
# Perform train-test division
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=test_size, 
                                                    random_state=ran_seed)

##############################################################################
# Training Process
##############################################################################

# Extract features for the training set
M_train = extract_features(X_train)   
# Normalize    
scaler = StandardScaler().fit(M_train)
M_train_n = scaler.transform(M_train)
"""
# We use a Support Vector Machine with RBF kernel
best_clf = SVC(probability=True)
# Train model
best_clf.fit(M_train_n, y_train)
"""


clf = xgb.XGBClassifier()

# Define hyperparameters for grid search
param_grid = {
    'n_estimators': [50, 100, 200],  # You can adjust the values as needed
    'learning_rate': [0.01, 0.1, 0.2],  # You can adjust the values as needed
    'max_depth': [3, 4, 5],  # You can adjust the values as needed
    'subsample': [0.8, 1.0],  # You can adjust the values as needed
}


grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(M_train_n, y_train)

# Get the best model from grid search
best_clf = grid_search.best_estimator_

##############################################################################
# Evaluation Process
##############################################################################


# Extract features for the test set
M_test = extract_features(X_test)   
# Normalize
M_test_n = scaler.transform(M_test)

# Obtain predicted labels and scores (probabilities) according to model
y_pred = best_clf.predict(M_test_n)
y_scores = best_clf.predict_proba(M_test_n)[:, 1]

# Obtain ROC curve values (FPR, TPR)
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_scores)
# Get Area Under the Curve
auc_svm = roc_auc_score(y_test, y_scores)
print(auc_svm)
print('\n\n\n')
# Plot ROC curve (displaying AUC)
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic Curve - AUC = ' +
          str(np.round(auc_svm,3)))
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
