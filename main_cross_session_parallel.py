import os
import time
import json, pickle
import itertools
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV,  cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from neurodeckit.loaddata import Dataset_Left_Right_MI, Dataset_MI
from neurodeckit.machine_learning import FBCSP
from neurodeckit import DL_Classifier
from neurodeckit.machine_learning.rpa import RCT
from ctssp import CTSSP, TRCTSSP, SBL_CTSSP
from joblib import Memory
from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp
from collections import defaultdict


def split_data_for_cv(data, label, kf): 
    
    all_data = {}
    all_indices = {}
    for cv, (train_index, test_index) in enumerate(kf.split(data, label)):
        # 保存训练集和测试集的索引
        all_indices[cv] = {}
        all_indices[cv]['train_index'] = train_index
        all_indices[cv]['test_index'] = test_index 
        # 保存训练集和测试集的数据
        all_data[cv] = {}
        all_data[cv]['traindata'] = data[train_index]
        all_data[cv]['trainlabel'] = label[train_index]
        all_data[cv]['testdata'] = data[test_index]
        all_data[cv]['testlabel'] = label[test_index]  
    
    return all_data, all_indices

# 划分函数  
def chunk_list(lst, chunk_size):  
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]  

def models_list(model_name):

    nfilter = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    rho = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    
    # Set up models
    if model_name == 'CSP':
        model = make_pipeline(
            CTSSP(), 
            LDA(solver='lsqr', shrinkage='auto'),   
            memory=memory
            )
        model = GridSearchCV(model, param_grid={'ctssp__nfilter': nfilter} , cv=5, scoring='accuracy')
    elif model_name == 'TRCSP':
        model = make_pipeline(
            TRCTSSP(nfilter=nfilter), 
            LDA(solver='lsqr', shrinkage='auto'),
            memory=memory
            )
        model = GridSearchCV(model, param_grid={'trctssp__rho': rho, 'trctssp__nfilter': nfilter}, cv=5, scoring='accuracy')
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
    elif model_name == 'CTSSP':
        model = make_pipeline(
            CTSSP(t_win=t_win, tau=tau),
            LDA(solver='lsqr', shrinkage='auto'),
            memory=memory
            )
        model = GridSearchCV(model, param_grid={'ctssp__nfilter': nfilter}, cv=5, scoring='accuracy')
    elif model_name == 'TRCSSP':
        model = make_pipeline(
            TRCTSSP(tau=tau),
            LDA(solver='lsqr', shrinkage='auto'),
            memory=memory
            )
        model = GridSearchCV(model, param_grid={'trctssp__rho': rho, 'trctssp__nfilter': nfilter}, cv=5, scoring='accuracy')
    elif model_name == 'TRCTSSP':    
        model = make_pipeline(
            TRCTSSP(t_win=t_win, tau=tau),
            LDA(solver='lsqr', shrinkage='auto'),
            memory=memory
            )
        model = GridSearchCV(model, param_grid={'trctssp__rho': rho, 'trctssp__nfilter': nfilter}, cv=5, scoring='accuracy')
    elif model_name == 'SBL-CSSP':
        model = SBL_CTSSP(tau=tau)
    elif model_name == 'SBL-CTSSP':
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
    # 检查计算任务是否已经完成 
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
        
        # # save accuracy and time results to json file
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
        
        # save model to pickle file
        if model_name in ml_methods:
            model_file_name = os.path.join(models_path, f"sub{sub:02d}_{cv}_{model_name}.pkl")
            with open(model_file_name, 'wb') as f:
                pickle.dump(model, f)
        
    except Exception as e:
        print(f"Model {model_name} failed: {e}")

if __name__ == '__main__':
    manager = mp.Manager()
    lock = manager.Lock() 

    # set up dataset
    # "BNCI2014_001", "BNCI2015_001", "Lee2019_MI", "Pan2023"
    # "BNCI2014_002", "BNCI2014_004", "Cho2017"
    # "BNCI2014_001", "Lee2019_MI", "Pan2023", "BNCI2014_002", "Cho2017"
    dataset_name = "BNCI2014_001"

    if dataset_name in ['Lee2019_MI', 'Cho2017']:
        channels = ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
                    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
                    ]
    else:
        channels = None
    
    fs = 160 if dataset_name in ['Cho2017', 'PhysionetMI'] else 128
    # datapath = r'E:\CoreyLin\小论文2023会议\数据处理python\datasets'
    datapath = r'E:\工作进展\小论文2023会议\数据处理python\datasets'
    
    if dataset_name in ["BNCI2014_001", "Lee2019_MI", "Cho2017", "Pan2023", "Shin2017A", "Weibo2014", "PhysionetMI"]:
        dataset = Dataset_Left_Right_MI(dataset_name,fs=fs,fmin=8,fmax=32,tmin=0,tmax=4,channels=channels,path=datapath)
    else:
        dataset = Dataset_MI(dataset_name,fs=fs,fmin=8,fmax=32,tmin=0,tmax=4,channels=channels,path=datapath)
    subject_list = dataset.subject_list

    # set up memory cache
    loc_path = 'E:/catch_temp'
    memory = Memory(location=loc_path, verbose=0, bytes_limit=1024*1024*1024*20)

    # set up results path
    results_path = os.path.join('Results3', dataset_name, 'cross_session')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # Set up models path
    models_path = os.path.join('F:/CTSSP_files/Models3', dataset_name, 'cross_session')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    # Set up models to run
    ml_methods = ['SBL-CSSP', 'SBL-CTSSP', 'CSP', 'TRCSP', 'FBCSP', 
                  'CSSP', 'CTSSP', 'TRCSSP', 'TRCTSSP', ]
    # dl_methods = ['EEGNet', 'sCNN', 'dCNN', 'Tensor-CSPNet', 'LMDA-Net']
    # methods = ml_methods + dl_methods
    methods = ml_methods

    # Set up CTSSP parameters
    tau = [0, 1]
    if dataset_name in ['Cho2017']:
        t_win = [(0, 2), (0.5, 2.5), (1, 3)]
    else:
        t_win = [(0, 3), (0.5, 3.5), (1, 4)]
    t_win = [(int(t*fs), int((w)*fs)) for t, w in t_win]

    # Set up cross-validation
    cv = 0
    
    # Set up n_jobs
    njobs = 1
    
    # Set up subjects
    chunked_lists = chunk_list(subject_list, 15)  
    
    for subjects in chunked_lists:
        
        # Set up all runs
        allruns = list(itertools.product(subjects, methods))

        all_data = defaultdict(lambda: defaultdict(lambda: {
            'traindata': [], 
            'trainlabel': [], 
            'testdata': [],
            'testlabel': []
            }))
        
        for sub in subjects:

            # load data and split into source and target domains
            data, label, info = dataset.get_data([sub])
            session_values = info['session'].unique()
            assert len(session_values) >= 2, "Subject should have at least two sessions."
            session_indices = info.groupby('session').apply(lambda x: x.index.tolist())
            session_index_dict = dict(zip(session_values, session_indices))

            all_data[sub][cv]['traindata'] = data[session_index_dict[session_values[-2]]]
            all_data[sub][cv]['trainlabel'] = label[session_index_dict[session_values[-2]]]
            all_data[sub][cv]['testdata'] = data[session_index_dict[session_values[-1]]]
            all_data[sub][cv]['testlabel'] = label[session_index_dict[session_values[-1]]]
            
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
                    ) for sub, model_name in allruns)
        

        