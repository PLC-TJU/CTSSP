"""
This code is used to train and test CTSSP and other models on the within-subject dataset.

The code uses joblib to parallelize the training and testing process on multiple CPUs.
The code uses the memory module to cache the covariance matrices and other intermediate results to save time.
The code uses the DL_Classifier class from the neurodeckit package to train and test deep learning models.  
The code uses the Dataset_Left_Right_MI and Dataset_MI classes from the neurodeckit package to load the dataset.
The code saves the accuracy and time results to a json file for each cross-validation fold and model.

The code can be run by setting the following variables:
- dataset_name: the name of the dataset to use
- model_name: the name of the model to use
- n_jobs: the number of CPUs to use for parallelization
- n_splits: the number of cross-validation folds to use
- n_repeats: the number of times to repeat the cross-validation
- fs: the sampling frequency of the EEG data
- tau: the time delay of the CTSSP filter
- t_win: the time window of the CTSSP filter
- results_path: the path to save the accuracy and time results to a json file

Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/10/02
License: All rights reserved
"""

import os
import time
import json
import itertools
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from neurodeckit.loaddata import Dataset_Left_Right_MI, Dataset_MI
from neurodeckit.machine_learning import FBCSP
from neurodeckit import DL_Classifier
from neurodeckit.machine_learning.rpa import RCT
from ctssp2 import CTSSP, SBL_CTSSP
from joblib import Memory
from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp


def split_data_for_cv(data, label, kf): 
    
    all_data = {}
    all_indices = {}
    for cv, (train_index, test_index) in enumerate(kf.split(data, label)):
        all_indices[cv] = {}
        all_indices[cv]['train_index'] = train_index
        all_indices[cv]['test_index'] = test_index 
        all_data[cv] = {}
        all_data[cv]['traindata'] = data[train_index]
        all_data[cv]['trainlabel'] = label[train_index]
        all_data[cv]['testdata'] = data[test_index]
        all_data[cv]['testlabel'] = label[test_index]  
    
    return all_data, all_indices
 
def chunk_list(lst, chunk_size):  
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]  

def models_list(model_name):

    nfilter = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Set up models
    if model_name == 'CSP':
        model = make_pipeline(
            CTSSP(), 
            LDA(solver='lsqr', shrinkage='auto'),   
            memory=memory
            )
        model = GridSearchCV(model, param_grid={'ctssp__nfilter': nfilter} , cv=5, scoring='accuracy')
    elif model_name == 'FBCSP':
        banks=[(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)]
        model = make_pipeline(
            FBCSP(fs=fs, banks=banks, n_components_select=10),
            LDA(solver='lsqr', shrinkage='auto'),
            memory=memory
            )
        model = GridSearchCV(model, param_grid={'fbcsp__nfilter': nfilter}, cv=5, scoring='accuracy')
    elif model_name == 'CSSP':
        model = make_pipeline(
            CTSSP(tau=tau),
            LDA(solver='lsqr', shrinkage='auto'),
            memory=memory
            )
        model = GridSearchCV(model, param_grid={'ctssp__nfilter': nfilter}, cv=5, scoring='accuracy')
    elif model_name == 'SBLEST':
        model = SBL_CTSSP(tau=tau)
    elif model_name == 'CTSSP':
        model = SBL_CTSSP(t_win=t_win, tau=tau)
    elif model_name == 'EEGNet':
        net = DL_Classifier(
        model_name='EEGNet', n_classes=2, fs=fs, batch_size=32, lr=0.001, max_epochs=300, 
        device='cuda')
        model = make_pipeline(
            RCT(mean_method='euclid'),
            net,
            memory=memory
            )
    elif model_name == 'sCNN':
        net= DL_Classifier(
        model_name='ShallowFBCSPNet', n_classes=2, fs=fs, batch_size=32, lr=0.001, max_epochs=300, 
        device='cuda')
        model = make_pipeline(
            RCT(mean_method='euclid'),
            net,
            memory=memory
            )
    elif model_name == 'dCNN':
        net = DL_Classifier(
        model_name='Deep4Net', n_classes=2, fs=fs, batch_size=32, lr=0.001, max_epochs=300, 
        device='cuda')    
        model = make_pipeline(
            RCT(mean_method='euclid'),
            net,
            memory=memory
            )
    elif model_name == 'Tensor-CSPNet':    
        net = DL_Classifier(
        model_name='Tensor_CSPNet', n_classes=2, fs=fs, batch_size=32, lr=0.001, max_epochs=100, 
        device='cuda')  
        model = make_pipeline(
            RCT(mean_method='euclid'),
            net,
            memory=memory
            )
    elif model_name == 'LMDA-Net':  
        net = DL_Classifier(
        model_name='LMDANet', n_classes=2, fs=fs, batch_size=32, lr=0.001, max_epochs=300, 
        device='cuda')   
        model = make_pipeline(
            RCT(mean_method='euclid'),
            net,
            memory=memory    
            )
    else:
        raise ValueError(f"Model {model_name} not found.")
    return model
        
def classification_within_subject(sub, cv, model_name, X_train, y_train, X_test, y_test):
    # check if results already exist for this cv and model
    file_name = os.path.join(results_path, f"sub{sub:02d}.json")   
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            results = [json.loads(line) for line in f.readlines() if line.strip()]
        for result in results:
            if result['cv'] == cv and result['model_name'] == model_name:
                print(f"Model {model_name} for cv={cv} already computed.")
                return

    model = models_list(model_name)
    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        start_time = time.time()
        score = model.score(X_test, y_test)
        test_time = time.time() - start_time
        
        # save accuracy and time results to json file
        result = {
            'cv': cv,
            'model_name': model_name, 
            'score': score * 100,
            'train_time': train_time, 
            'test_time': test_time
            }
        with lock:
            with open(file_name, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
    except Exception as e:
        print(f"Model {model_name} failed: {e}")

if __name__ == '__main__':
    manager = mp.Manager()
    lock = manager.Lock() 

    # set up dataset
    # "BNCI2014_001", "Lee2019_MI", "Pan2023", "BNCI2014_002", "Cho2017"
    dataset_name = "BNCI2014_001"

    if dataset_name in ['Lee2019_MI', 'Cho2017']:
        channels = ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
                    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
                    ] # selected recommended channels 
    else:
        channels = None
    
    fs = 160 if dataset_name in ['Cho2017', 'PhysionetMI'] else 128
    datapath = None # set to None if dataset is not downloaded
    
    if dataset_name in ["BNCI2014_001", "Lee2019_MI", "Cho2017", "Pan2023", "Shin2017A", "Weibo2014", "PhysionetMI"]:
        dataset = Dataset_Left_Right_MI(dataset_name,fs=fs,fmin=8,fmax=32,tmin=0,tmax=4,channels=channels,path=datapath)
    else:
        dataset = Dataset_MI(dataset_name,fs=fs,fmin=8,fmax=32,tmin=0,tmax=4,channels=channels,path=datapath)
    subject_list = dataset.subject_list
    
    # set up memory cache
    loc_path = 'catch_temp'
    memory = Memory(location=loc_path, verbose=0)

    # set up results path
    results_path = os.path.join('Results', dataset_name, 'within_subject')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # Set up models to run
    ml_methods = ['CSP', 'CSSP', 'FBCSP', 'SBLEST', 'CTSSP']
    dl_methods = ['EEGNet', 'sCNN', 'dCNN', 'Tensor-CSPNet', 'LMDA-Net']
    methods = ml_methods + dl_methods
    methods = ml_methods   

    # Set up CTSSP parameters
    tau = [0, 1]
    if dataset_name in ['Cho2017']:
        t_win = [(0, 2), (0.5, 2.5), (1, 3)]
    else:
        t_win = [(0, 3), (0.5, 3.5), (1, 4)]
    t_win = [(int(t*fs), int((w)*fs)) for t, w in t_win]

    # Set up cross-validation
    n_splits = 10
    n_repeats = 5
    kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=2024)

    # Set up n_jobs
    njobs = 1 # for deeplearning models, set to 1 to avoid memory issues
    
    # Set up subjects
    chunked_lists = chunk_list(subject_list, 4) # multiprocessing with 4 subjects at a time
    
    for subjects in chunked_lists:
    
        # Set up all runs
        allruns = list(itertools.product(subjects, range(n_splits*n_repeats), methods))
        
        all_data = {}
        for sub in subjects:

            # load data and split into source and target domains
            data, label, info = dataset.get_data([sub])
            
            # split data into training and testing sets
            all_data[sub], _ = split_data_for_cv(data, label, kf)
            
        # run models for each split and each method
        with parallel_backend('loky', n_jobs=njobs):
            Parallel(batch_size=1, verbose=len(allruns))(
                delayed(classification_within_subject)(
                    sub,
                    cv,
                    model_name,
                    all_data[sub][cv]['traindata'], 
                    all_data[sub][cv]['trainlabel'], 
                    all_data[sub][cv]['testdata'], 
                    all_data[sub][cv]['testlabel'], 
                    ) for sub, cv, model_name in allruns)
        

        