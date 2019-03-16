# -*- coding: utf-8 -*-
from __future__ import print_function
import torch, os, time, warnings, codecs
from collections import defaultdict
from mpi4py import MPI




from Hydra.Models import ModelConvDecon
from Hydra.LoadingCorpus import Loading_and_formating
from Hydra.Training_and_Validation import Training_and_Validation
from Hydra.Utils import get_param_dict, train_iteretions, tagging, load_tools,\
check_gpu_and_select_workers, preprocessing_corpus_by_worker_no_O, start_mpi, arguments_and_configuration





'''seed for torch'''
torch.manual_seed(1)
torch.backends.cudnn.benchmark=True
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


'''Initialise MPI communication '''
comm, size, rank, name_host = start_mpi()

'''read the params file'''
params = arguments_and_configuration()

'''start to measure time for worktime estimation'''
start_time = time.time()

'''
check and compare number of processes and available GPUs
and
create a list of active workers with GPU.
'''
active_workers, use_cuda, extra_check = check_gpu_and_select_workers(name_host=name_host, gpu=params['gpu'],
                                                                     rank=rank, size=size, comm=comm)

'''preprocessing the corpora is being done just by process no.0'''
if rank == 0:
    char_vector_dict, char_idx, number_char, name_of_train_frag, name_of_val_frag,\
    name_of_test_frag, models_path, train_stats_path, corpus_stats_path\
    = preprocessing_corpus_by_worker_no_O(params=params, active_workers=active_workers, comm=comm)
   
'''only a process being an active worker after sanity check can proceed'''    
if rank in active_workers and extra_check:
    
    if params['training_mode']:
    
        
        '''other processes are waiting for information from process no.0'''
        if rank != 0:
            char_vector_dict = comm.recv(source=0, tag=111)
            char_idx = comm.recv(source=0, tag=111)
            number_char = comm.recv(source=0, tag=111)
            name_of_train_frag = comm.recv(source=0, tag=111)
            name_of_val_frag = comm.recv(source=0, tag=111)
            name_of_test_frag = comm.recv(source=0, tag=111)
            models_path = comm.recv(source=0, tag=111)
            train_stats_path = comm.recv(source=0, tag=111)
            corpus_stats_path = comm.recv(source=0, tag=111)
        time.sleep(2)
        
        encoder_lemma = None
        encoder_pos = None
        encoder_morph = None
        
        lemma_size = 1
        pos_size = 1
        morph_size = 1
        
        dictionary = None
        if params['dictionary_folder']:
            
            imported_and_prepared_dict = Loading_and_formating(
                    directory=params['dictionary_folder'],
                    nb_instances=params['nb_instances'],
                    include_lemma=params['include_lemma'],
                    include_morph=params['include_morph'],
                    include_pos=params['include_pos'],
                    v2u=params['v2u']
                                    )
            
            dictionary = imported_and_prepared_dict.load_corpus_folder()
            
            
            #for line in codecs.open(params['lemma_list'], 'r', encoding='utf8'):
            # print(line)
            #    line = line.strip()
            #    line = line.replace('v', 'u')
            #    lemma_list.append(line)
        #print(lemma_list)
        #quit() 
        encoders_path = os.path.join(params['cellarium_folder'],'Encoders')   
        encoder_letters = load_tools(os.path.join(encoders_path,'encoder_letters'))
        letters_size = len(encoder_letters.classes_)
        letters_pad = encoder_letters.transform(['<PAD>'])[0]
        if params['include_lemma']:
            encoder_lemma = load_tools(os.path.join(encoders_path,'encoder_lemma'))
            lemma_size = len(encoder_lemma.classes_)
        
        if params['include_pos']:
            encoder_pos = load_tools(os.path.join(encoders_path,'encoder_pos'))
            pos_size = len(encoder_pos.classes_)
        if params['include_morph']:
            encoder_morph = load_tools(os.path.join(encoders_path,'encoder_morph'))
            morph_size = len(encoder_morph.classes_)
        
        
        
        max_len_tok = params['max_len_tok']
        
        embedder_lemma, embedder_pos, embedder_morph = None, None, None
        
        
        
        model = ModelConvDecon(embedder_size=params['embedder_letters_hidden_size'],
                          hidden_size_model=params['hidden_size_model'],
                         include_lemma = params['include_lemma'], include_pos = params['include_pos'], include_morph = params['include_morph'],
                         max_len_tok= params['max_len_tok'],
                         letters_size=letters_size,
                         lemma_size= lemma_size, pos_size=pos_size, morph_size=morph_size,
                         letters_pad=letters_pad,
                         context_size=params['context_size'],
                         max_len_lemma=params['max_len_lemma'],
                         max_tag_len_seq=params['max_tag_len_seq'],
                         use_cuda=use_cuda,
                         nb_kernels_token = params['nb_kernels_token'], nb_final_kernels_token = params['nb_final_kernels_token'], 
                         nb_kernels_minic_L = params['nb_kernels_minic_l'], nb_final_kernels_minic_L = params['nb_final_kernels_minic_l'],
                         nb_kernels_minic_R = params['nb_kernels_minic_r'], nb_final_kernels_minic_R = params['nb_final_kernels_minic_r'],
                         nb_kernels_c_L = params['nb_kernels_c_l'], nb_final_kernels_c_L = params['nb_final_kernels_c_l'],
                         nb_kernels_c_R = params['nb_kernels_c_r'], nb_final_kernels_c_R = params['nb_final_kernels_c_r']
                         )
        
        if use_cuda:
            model.cuda()
        
        model_optimizer = torch.optim.Adam(model.parameters(),
                                   lr=params['lr'], 
                                   eps=0.001
                                   )
    
        
        
        
        
        try:
            model.load_state_dict(torch.load(os.path.join(models_path, "model_saved_proc_{}".format(rank))))
            print('loaded a partially trained model no.{}'.format(rank))
        except:
            print('creating a new model no.{}'.format(rank))
            pass
        
        try:
            model_optimizer.load_state_dict(torch.load(os.path.join(models_path, "optimiser_saved_proc_{}".format(rank))))
            print('loaded a model optimiser no.{}'.format(rank))
        except:
            print('creating a new model optimiser no.{}'.format(rank))
            pass
        
        
        criterion = torch.nn.NLLLoss()
        
        name_of_assigned_corpus_file = str(name_of_train_frag) + '_' + str(rank) + '.txt'
        corpus_fragment = codecs.open(name_of_assigned_corpus_file, 'r', encoding='utf8')
        
        name_of_assigned_dev_file = str(name_of_val_frag) + '.txt'
        dev_fragment = codecs.open(name_of_assigned_dev_file, 'r', encoding='utf8')
        
        name_of_assigned_test_file = str(name_of_test_frag) + '.txt'
        test_fragment = codecs.open(name_of_assigned_test_file, 'r', encoding='utf8')
        
        
        
        print('process no.{}: loaded file {}'.format(rank, name_of_assigned_corpus_file))
        
        try:
            with open(os.path.join(models_path, 'num_epochs_acc_proc_{}.txt'.format(rank)), "r") as myfile:
                n_iters_acc= myfile.read()
                n_iters = params['number_epochs'] - int(n_iters_acc)
            print('{} epochs to finish'.format(n_iters))
        except Exception as e:
            n_iters = params['number_epochs']
            print('the model will be trained for {} epochs'.format(n_iters))
        
        
        '''Initialise an alpha variable for the gossip protocol'''
        alpha = 1.0 / len(active_workers)
        
        
        train_and_val = Training_and_Validation(
                train_fragment=corpus_fragment,
                val_fragment=dev_fragment,
                test_fragment=test_fragment,
                model=model,
                model_optimizer=model_optimizer,
                criterion=criterion,
                char_vector_dict=char_vector_dict,
                include_lemma=params['include_lemma'],
                include_morph=params['include_morph'],
                include_pos=params['include_pos'],
                encoder_lemma=encoder_lemma,
                encoder_letters=encoder_letters,
                encoder_pos=encoder_pos,
                encoder_morph=encoder_morph,
                dictionary=dictionary,
                max_len_tok=params['max_len_tok'],
                max_len_lemma = params['max_len_lemma'],
                max_tag_len_seq = params['max_tag_len_seq'],
                batch_size = params['batch_size'],
                print_freq= params['print_freq'],
                context_size = params['context_size'],
                alpha=alpha,
                rank=rank,
                size=size,
                active_workers=active_workers,
                use_cuda = use_cuda,
                p_value=params['p_value'],
                cellarium_folder = params['cellarium_folder'],
                models_folder = models_path,
                corpus_stats_folder = corpus_stats_path,
                n_iters = n_iters
                )
        
        train_iteretions(train_and_val=train_and_val,
                         train_stats_folder= train_stats_path,
                         n_iters = n_iters,
                         lr_mod_param = params['lr_mod_param'],
                         lr_mod_moment = params['lr_mod_moment'],
                         n_val = params['n_val'],
                         rank = rank,
                         active_workers=active_workers,
                         comm = comm,
                         use_cuda = use_cuda,
                         start_time = start_time
                         )


    if params['tagging_mode']:
        if rank == 0:
            msg = tagging(params)
            print(msg)

if rank == 0:
    
    for target in range(size-1):
        target += 1
        msg = 'maltho thi afrio lito'
        comm.ssend(msg, dest = target, tag= 101)

else:
    
    msg = comm.recv(source=MPI.ANY_SOURCE, tag = 101)
    print(msg)
    
