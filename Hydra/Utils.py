import ConfigParser, math, time, os, pickle, torch, random, itertools, codecs, argparse
import numpy as np
from fuzzywuzzy import fuzz
from collections import defaultdict
from torch.autograd import Variable
from Hydra.Models import ModelConvDecon
from Hydra.LoadingCorpus import Loading_and_formating

import gc
###

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

from mpi4py import MPI


#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#rank = comm.Get_rank()

#global use_cuda
#use_cuda = torch.cuda.is_available()

#global number_gpu
#number_gpu = torch.cuda.device_count()




'''two auxiliary function for reading data of a parameters file'''
def isInt(inputString):
    return all(char.isdigit() for char in inputString)
########################################################
def isFloat(inputString):
    return all(char.isdigit() or char == '.' for char in inputString)
########################################################



'''a function for creating a dictionary of parameters'''
def get_param_dict(p):
    config = ConfigParser.ConfigParser()
    config.read(p)
    # parse the param
    param_dict = dict()
    for section in config.sections():
        for name, value in config.items(section):
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            elif isInt(value):
                value = int(value)
            elif isFloat(value):
                value = float(value)
            param_dict[name] = value
    return param_dict


###############################################################################
'''time measurment modules for optional debugging/development'''
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s ( %s)' % (asMinutes(s), asMinutes(rs))

################################################################################


'''
A function for parsing arguments and/or reading the configuration file
'''

def arguments_and_configuration():
    parser = argparse.ArgumentParser(description=
                                     '''
    Hydra.
    Integrated Tagger-Lemmatiser with Deep Learning and Parallel Computing.
    Hydra is a universal, language-independent tagger-lemmatiser capable of training language models 
    from annotated corpora and efficiently tagging new texts both within one's own Python script or 
    via the command line by creating new text files for further processing outside Python environment.
                                     '''
                                     
                                     )
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to your configuration file')
    parser.add_argument('--train_directory', type=str, default=None,
                        help='Path to the training directory')
    parser.add_argument('--dev_directory', type=str, default=None,
                        help='Path to the development directory')
    parser.add_argument('--test_directory', type=str, default=None,
                        help='Path to the test directory')
    parser.add_argument('--texts_to_tag_directory', type=str, default=None,
                        help='Path to the directory of texts-to-tag')
    parser.add_argument('--texts_tagged_directory', type=str, default=None,
                        help='Provide a name of the directory for tagged texts')
    parser.add_argument('--training_mode', type=bool, default=True,
                        help="'True' to start the training, 'False' to omit the training phase")
    parser.add_argument('--tagging_mode', type=bool, default=True,
                        help="'True' to tag texts, otherwise 'False'")
    parser.add_argument('--gpu', type=bool, default=True,
                        help="'True' to enable the GPU-based training, otherwise 'False'")
    parser.add_argument('--dictionary_folder', type=str, default=None,
                        help="the path to the dictionary directory")
    parser.add_argument('--nb_instances', type=int, default=None,
                        help="the path to the dictionary directory")
    parser.add_argument('--include_lemma', type=bool, default=True,
                        help="'True' to include lemmas in the model training or in the tagging mode")
    parser.add_argument('--include_pos', type=bool, default=True,
                        help="'True' to include POS in the model training or in the tagging mode")
    parser.add_argument('--include_morph', type=bool, default=True,
                        help="'True' to include morphology tags in the model training or in the tagging mode")
    parser.add_argument('--nb_kernels_token', type=int, default=100,
                        help="The number of convolving kernels for the token branch")
    parser.add_argument('--nb_final_kernels_token', type=int, default=100,
                        help="The number of convolving kernels for the token branch in the final layer")
    parser.add_argument('--nb_kernels_minic_l', type=int, default=100,
                        help="The number of convolving kernels for the left mini-context")
    parser.add_argument('--nb_final_kernels_minic_l', type=int, default=100,
                        help="The number of convolving kernels for the left mini-context in the final layer")
    parser.add_argument('--nb_kernels_minic_r', type=int, default=100,
                        help="The number of convolving kernels for the right mini-context")
    parser.add_argument('--nb_final_kernels_minic_r', type=int, default=100,
                        help="The number of convolving kernels for the right mini-context in the final layer")
    parser.add_argument('--nb_kernels_c_l', type=int, default=100,
                        help="The number of convolving kernels for the left context")
    parser.add_argument('--nb_final_kernels_c_l', type=int, default=100,
                        help="The number of convolving kernels for the left context in the final layer")
    parser.add_argument('--nb_kernels_c_r', type=int, default=100,
                        help="The number of convolving kernels for the right context")
    parser.add_argument('--nb_final_kernels_c_r', type=int, default=100,
                        help="The number of convolving kernels for the right context in the final layer")
    parser.add_argument('--v2u', type=bool, default=True,
                        help="'True' for casting 'v' into 'u', otherwise 'False'")
    parser.add_argument('--min_lem_cnt', type=int, default=1,
                        help="The minimal number of lemma occurrences")
    parser.add_argument('--min_tok_cnt', type=int, default=1,
                        help="The minimal number of token occurrences")
    parser.add_argument('--max_len_lemma', type=int, default=10,
                        help="The maximal length of produced lemmas")
    parser.add_argument('--max_seq_len', type=int, default=3,
                        help="The maximal length of the context sequence")
    parser.add_argument('--max_tag_len_seq', type=int, default=3,
                        help="The maximal length of the tag sequence")
    parser.add_argument('--name_of_train_frag', type=str, default='train_frag',
                        help="The name of files with training fragments")
    parser.add_argument('--name_of_val_frag', type=str,default='val_frag',
                        help="The name of files with validation fragments")
    parser.add_argument('--name_of_test_frag', type=str,default='test_frag',
                        help="The name of files with test fragments")
    parser.add_argument('--cellarium_folder', type=str, default='cellarium1',
                        help="The name of the directory for all produced data files")
    parser.add_argument('--print_freq', type=float, default=1.0,
                        help=
                        """
                        Printing frequency: 1.0 for one single printout per epoch, 
                        1/n for each batch with n = number of epochs
                        """)
    parser.add_argument('--batch_size', type=int, default=30,
                        help="The batch size being the number of tokens processed with their respective contexts at once")
    parser.add_argument('--context_size', type=int, default=3,
                        help="The context size")
    parser.add_argument('--max_len_tok', type=int, default=15,
                        help="The maximal length of the token")
    parser.add_argument('--embedder_letters_hidden_size', type=int, default=32,
                        help="The size of letter embbedings")
    parser.add_argument('--hidden_size_model', type=int, default=32,
                        help="The size of hidden layers of the model")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="The learning rate")
    parser.add_argument('--lr_mod_param', type=float, default=0.5,
                        help="The ratio of decreasing of the learning rate after a fixed number of epochs")
    parser.add_argument('--lr_mod_moment', type=bool, default=5,
                        help="The number of epochs after which the learning rate is decreased")
    parser.add_argument('--p_value', type=float, default=0.01,
                        help="The frequency rate of exchange between workers each n batches")
    parser.add_argument('--number_epochs', type=int, default=100,
                        help="The number epochs for training")
    parser.add_argument('--n_val', type=int, default=100,
                        help="The frequency ratio for validation after n epochs")
    
    args = parser.parse_args()
    
    '''read parameters values from a configuration file...'''
    if args.config_file:
        params = get_param_dict(args.config_file)
    else:
        params = {}
    '''... or from command line inputs'''
    params['train_directory'] = args.train_directory if not \
    args.train_directory == parser.get_default('train_directory') else params['train_directory']
    params['dev_directory'] = args.dev_directory if not \
    args.dev_directory == parser.get_default('dev_directory') else params['dev_directory']
    params['test_directory'] = args.test_directory if not \
    args.test_directory == parser.get_default('test_directory') else params['test_directory']
    params['texts_to_tag_directory'] = args.texts_to_tag_directory if not \
    args.texts_to_tag_directory == parser.get_default('texts_to_tag_directory') else params['texts_to_tag_directory']
    params['texts_tagged_directory'] = args.texts_tagged_directory if not \
    args.texts_tagged_directory == parser.get_default('texts_tagged_directory') else params['texts_tagged_directory']
    params['training_mode'] = args.training_mode if not \
    args.training_mode == parser.get_default('training_mode') else params['training_mode']
    params['tagging_mode'] = args.tagging_mode if not \
    args.tagging_mode == parser.get_default('tagging_mode') else params['tagging_mode']
    params['gpu'] = args.gpu if not \
    args.gpu == parser.get_default('gpu') else params['gpu']
    params['dictionary_folder'] = args.dictionary_folder if not \
    args.dictionary_folder == parser.get_default('dictionary_folder') else params['dictionary_folder']
    params['nb_instances'] = args.nb_instances if not \
    args.nb_instances == parser.get_default('nb_instances') else params['nb_instances']
    params['include_lemma'] = args.include_lemma if not \
    args.include_lemma == parser.get_default('include_lemma') else params['include_lemma']
    params['include_pos'] = args.include_pos if not \
    args.include_pos == parser.get_default('include_pos') else params['include_pos']
    params['include_morph'] = args.include_morph if not \
    args.include_morph == parser.get_default('include_morph') else params['include_morph']
    params['nb_kernels_token'] = args.nb_kernels_token if not \
    args.nb_kernels_token == parser.get_default('nb_kernels_token') else params['nb_kernels_token']
    params['nb_final_kernels_token'] = args.nb_final_kernels_token if not \
    args.nb_final_kernels_token == parser.get_default('nb_final_kernels_token') else params['nb_final_kernels_token']
    params['nb_kernels_minic_l'] = args.nb_kernels_minic_l if not \
    args.nb_kernels_minic_l == parser.get_default('nb_kernels_minic_l') else params['nb_kernels_minic_l']
    params['nb_final_kernels_minic_l'] = args.nb_final_kernels_minic_l if not \
    args.nb_final_kernels_minic_l == parser.get_default('nb_final_kernels_minic_l') else params['nb_final_kernels_minic_l']
    params['nb_kernels_minic_r'] = args.nb_kernels_minic_r if not \
    args.nb_kernels_minic_r == parser.get_default('nb_kernels_minic_r') else params['nb_kernels_minic_r']
    params['nb_final_kernels_minic_r'] = args.nb_final_kernels_minic_r if not \
    args.nb_final_kernels_minic_r == parser.get_default('nb_final_kernels_minic_r') else params['nb_final_kernels_minic_r']
    params['nb_kernels_c_l'] = args.nb_kernels_c_l if not \
    args.nb_kernels_c_l == parser.get_default('nb_kernels_c_l') else params['nb_kernels_c_l']
    params['nb_final_kernels_c_l'] = args.nb_final_kernels_c_l if not \
    args.nb_final_kernels_c_l == parser.get_default('nb_final_kernels_c_l') else params['nb_final_kernels_c_l']
    params['nb_kernels_c_r'] = args.nb_kernels_c_r if not \
    args.nb_kernels_c_r == parser.get_default('nb_kernels_c_r') else params['nb_kernels_c_r']
    params['nb_final_kernels_c_r'] = args.nb_final_kernels_c_r if not \
    args.nb_final_kernels_c_r == parser.get_default('nb_final_kernels_c_r') else params['nb_final_kernels_c_r']
    params['v2u'] = args.v2u if not \
    args.v2u == parser.get_default('v2u') else params['v2u']
    params['min_lem_cnt'] = args.min_lem_cnt if not \
    args.min_lem_cnt == parser.get_default('min_lem_cnt') else params['min_lem_cnt']
    params['min_tok_cnt'] = args.min_tok_cnt if not \
    args.min_tok_cnt == parser.get_default('min_tok_cnt') else params['min_tok_cnt']
    params['max_len_lemma'] = args.max_len_lemma if not \
    args.max_len_lemma == parser.get_default('max_len_lemma') else params['max_len_lemma']
    params['max_seq_len'] = args.max_seq_len if not \
    args.max_seq_len == parser.get_default('max_seq_len') else params['max_seq_len']
    params['max_tag_len_seq'] = args.max_tag_len_seq if not \
    args.max_tag_len_seq == parser.get_default('max_tag_len_seq') else params['max_tag_len_seq']
    params['name_of_train_frag'] = args.name_of_train_frag if not \
    args.name_of_train_frag == parser.get_default('name_of_train_frag') else params['name_of_train_frag']
    params['name_of_val_frag'] = args.name_of_val_frag if not \
    args.name_of_val_frag == parser.get_default('name_of_val_frag') else params['name_of_val_frag']
    params['name_of_test_frag'] = args.name_of_test_frag if not \
    args.name_of_test_frag == parser.get_default('name_of_test_frag') else params['name_of_test_frag']
    params['cellarium_folder'] = args.cellarium_folder if not \
    args.cellarium_folder == parser.get_default('cellarium_folder') else params['cellarium_folder']
    params['print_freq'] = args.print_freq if not \
    args.print_freq == parser.get_default('print_freq') else params['print_freq']
    params['batch_size'] = args.batch_size if not \
    args.batch_size == parser.get_default('batch_size') else params['batch_size']
    params['context_size'] = args.context_size if not \
    args.context_size == parser.get_default('context_size') else params['context_size']
    params['max_len_tok'] = args.max_len_tok if not \
    args.max_len_tok == parser.get_default('max_len_tok') else params['max_len_tok']
    params['embedder_letters_hidden_size'] = args.embedder_letters_hidden_size if not \
    args.embedder_letters_hidden_size == parser.get_default('embedder_letters_hidden_size') else params['embedder_letters_hidden_size']
    params['hidden_size_model'] = args.hidden_size_model if not \
    args.hidden_size_model == parser.get_default('hidden_size_model') else params['hidden_size_model']
    params['lr'] = args.lr if not \
    args.lr == parser.get_default('lr') else params['lr']
    params['lr_mod_param'] = args.lr_mod_param if not \
    args.lr_mod_param == parser.get_default('lr_mod_param') else params['lr_mod_param']
    params['lr_mod_moment'] = args.lr_mod_moment if not \
    args.lr_mod_moment == parser.get_default('lr_mod_moment') else params['lr_mod_moment']
    params['p_value'] = args.p_value if not \
    args.p_value == parser.get_default('p_value') else params['p_value']
    params['number_epochs'] = args.number_epochs if not \
    args.number_epochs == parser.get_default('number_epochs') else params['number_epochs']
    params['n_val'] = args.n_val if not \
    args.n_val == parser.get_default('n_val') else params['n_val']
    
    ###rewrite it as for the following function: 
    #To get all defaults:
    #all_defaults = {}
    #for key in vars(args):
    #    all_defaults[key] = parser.get_default(key)
    
    #print(params['train_directory'])
    #quit()
    if params['training_mode'] == True and params['train_directory'] == None:
        raise ValueError("The name of a directory with training files has not been specified")
    if params['training_mode'] == True and params['dev_directory'] == None:
        raise Warning("""
        The name of a directory with data files 
        for the development phase has not been specified.
        Hydra will use the training directory instead.
        """)
    if params['training_mode'] == True and params['test_directory'] == None:
        raise Warning("""
        The name of a directory with data files 
        for the test phase has not been specified.
        Hydra will use the training directory instead.
        """)
    if params['tagging_mode'] == True and params['texts_to_tag_directory'] == None:
        raise ValueError("""
        The name of a directory with texts to tag 
        has not been specified.
        """)
    if params['tagging_mode'] == True and params['texts_tagged_directory'] == None:
        raise ValueError("""
        The name of a target directory for tagged texts
        has not been specified.
        """)
    #print(params)
    #quit()
    
    return params
    
    


'''
A function for initialising the mpi module
'''
def start_mpi():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name_host = MPI.Get_processor_name()
    
    return comm, size, rank, name_host

''' a function for pruning a text from non-alphanumeric symbols --- a copy from LoadingCorpus '''

def leave_only_alphanumeric(string):
    return ''.join(ch if ch.isalnum() else ' ' for ch in string)


'''
A function to check and compare number of processes and available GPUs
and
create a list of active workers with GPU.
'''
def check_gpu_and_select_workers(name_host=None, gpu=False, rank=0, size=0, comm=None):
    
    if gpu:
        '''
        process no. 0 must pair the number of processes to the number of GPUs per machine  
        '''
            
            
        '''define usage GPU and CUDA via PyTorch and the number of accessible GPUs'''
        
        use_cuda = torch.cuda.is_available()
        if use_cuda == False:
            raise Exception(
            '''You do not have access to gpu resources. 
            Check either your CUDA installation and compatibility
            or run Hydra once again with gpu-option set to 'False' providing
            a sufficent number of CPU cores.''')
        #global number_gpu
        number_gpu = torch.cuda.device_count()
        print('number gpus on process no. {}: {}'.format(rank, number_gpu))
        print('cuda in use: {}'.format(use_cuda))
            
        if rank !=0:
            msg = (name_host, rank, number_gpu)
            comm.ssend(msg, dest = 0, tag=102)
            
        if rank == 0:
            hosts_proc = defaultdict(list)
            hosts_proc[name_host].append(rank)
            hosts_gpu = defaultdict(list)
            hosts_gpu[name_host] = number_gpu
                
            for x in range(size-1):
                host, proc_rank, number_gpu = comm.recv(source=MPI.ANY_SOURCE, tag=102)
                hosts_proc[host].append(proc_rank)
                hosts_gpu[host] = number_gpu
                
            active_workers = []
                
            for host, proc in hosts_proc.iteritems():
                nb_gpus = hosts_gpu[host]
                active_workers.extend(proc[0:nb_gpus])
            
            for target in range(size-1):
                target += 1
                comm.send((active_workers, hosts_proc, hosts_gpu), dest = target, tag=103)
                #print('list of processes as active workers: {}'.format(active_workers))    
        else:
            active_workers, hosts_proc, hosts_gpu = comm.recv(source=MPI.ANY_SOURCE, tag=103)
            
        proc_list = hosts_proc[name_host]
        your_gpus = hosts_gpu[name_host]
        
    else:
        '''
        all available processes are made to workers (CPU-based training)
        '''
        use_cuda = False
        active_workers=range(size)
        
    print('list of processes as active workers: {}'.format(active_workers))   
        
    if gpu:
        your_position = [i for i,x in enumerate(proc_list) if x == rank][0]
        
        if rank in active_workers and your_position < your_gpus:
            torch.cuda.set_device(your_position)
            print('worker no.{} is working on device no.{}/{} on host {}'.format(rank,your_position, torch.cuda.device_count(), name_host))
    
    '''extra check in the case of the gpu training'''
    if gpu:
        if your_position < your_gpus:
            extra_check = True
        else:
            extra_check = False
    else:
        extra_check = True
    
    
    
    return active_workers, use_cuda, extra_check


def preprocessing_corpus_by_worker_no_O(params=None, active_workers=[], comm=None):
#if rank == 0:
    '''process no.0 creates and prepares necessary files'''
    if not os.path.exists(params['cellarium_folder']):
        os.makedirs(params['cellarium_folder'])
    path_to_corp_frag_folder = os.path.join(params['cellarium_folder'],'Corpus Fragments')
    if not os.path.exists(path_to_corp_frag_folder):
        os.makedirs(path_to_corp_frag_folder)
    models_path = os.path.join(params['cellarium_folder'],'Models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    train_stats_path = os.path.join(params['cellarium_folder'],'Training Stats')
    if not os.path.exists(train_stats_path):
        os.makedirs(train_stats_path)
    
    path_to_train_prep_corpus = os.path.join(path_to_corp_frag_folder, params['name_of_train_frag'])
    path_to_val_prep_corpus = os.path.join(path_to_corp_frag_folder, params['name_of_val_frag'])
    path_to_test_prep_corpus = os.path.join(path_to_corp_frag_folder, params['name_of_test_frag'])
    
    imported_and_prepared_corpus = Loading_and_formating(
                    directory=params['train_directory'],
                    nb_instances=params['nb_instances'],
                    include_lemma=params['include_lemma'],
                    include_morph=params['include_morph'],
                    include_pos=params['include_pos'],
                    v2u=params['v2u'],
                    min_lem_cnt=params['min_lem_cnt'],
                    min_tok_cnt=params['min_tok_cnt'],
                    num_splits=len(active_workers),
                    name_of_frag=path_to_train_prep_corpus
                                    )
    
    char_vector_dict, char_idx, number_char = imported_and_prepared_corpus.index_characters1()
    
    encoder_letters = imported_and_prepared_corpus.index_characters2()
    
    
    encoders_path = os.path.join(params['cellarium_folder'],'Encoders')
    if not os.path.exists(encoders_path):
        os.makedirs(encoders_path)
    
    save_tools( encoder_letters, os.path.join(encoders_path,'encoder_letters'))
    if params['include_lemma']:
        encoder_lemma = imported_and_prepared_corpus.index_lemmas()
        save_tools(encoder_lemma, os.path.join(encoders_path, 'encoder_lemma'))
    if params['include_pos']:
        encoder_pos = imported_and_prepared_corpus.index_pos()
        save_tools(encoder_pos, os.path.join(encoders_path, 'encoder_pos'))
    if params['include_morph']:
        encoder_morph = imported_and_prepared_corpus.index_morph()
        save_tools(encoder_morph, os.path.join(encoders_path, 'encoder_morph'))
    
    
    imported_and_prepared_corpus.create_fragments()
    
    imported_and_prepared_corpus.index_tokens()
    train_tokens_counted = imported_and_prepared_corpus.trunc_lems
    train_nb_multitokens = imported_and_prepared_corpus.nb_multitokens
    if params['include_lemma']:
        train_lemmas_counted = imported_and_prepared_corpus.lemmas_unique
        train_multilabels_lemmas_counted = imported_and_prepared_corpus.count_multilabel_lemma()
        #print(train_lemmas_counted)
        #print(set(train_multilabels_lemmas_counted))
        #print(set(train_lemmas_counted).intersection(train_multilabels_lemmas_counted))
        #quit()
    
    
    if params['dev_directory']:
    
        imported_and_prepared_dev = Loading_and_formating(
                        directory=params['dev_directory'],
                        nb_instances=params['nb_instances'],
                        include_lemma=params['include_lemma'],
                        include_morph=params['include_morph'],
                        include_pos=params['include_pos'],
                        v2u=params['v2u'],
                        min_lem_cnt=params['min_lem_cnt'],
                        min_tok_cnt=params['min_tok_cnt'],
                        num_splits=len(active_workers),
                        name_of_frag=path_to_val_prep_corpus
                                        )
        
        imported_and_prepared_dev.create_dev_file()
        
    
    if params['test_directory']:
        imported_and_prepared_test = Loading_and_formating(
                        directory=params['test_directory'],
                        nb_instances=params['nb_instances'],
                        include_lemma=params['include_lemma'],
                        include_morph=params['include_morph'],
                        include_pos=params['include_pos'],
                        v2u=params['v2u'],
                        min_lem_cnt=params['min_lem_cnt'],
                        min_tok_cnt=params['min_tok_cnt'],
                        num_splits=len(active_workers),
                        name_of_frag=path_to_test_prep_corpus
                                        )
        
        imported_and_prepared_test.create_dev_file()
    
    
    name_of_train_frag = imported_and_prepared_corpus.name_of_frag
    if params['dev_directory']:
        name_of_val_frag = imported_and_prepared_dev.name_of_frag
    else:
        name_of_val_frag = None
    if params['test_directory']:
        name_of_test_frag = imported_and_prepared_test.name_of_frag
    else:
        name_of_test_frag = None
   
    
    #########
    #statistics
    
    if params['dev_directory']:
        imported_and_prepared_dev.index_tokens()
        dev_tokens_counted = imported_and_prepared_dev.trunc_lems
        dev_nb_multitokens = imported_and_prepared_dev.nb_multitokens
    if params['test_directory']:
        imported_and_prepared_test.index_tokens()
        test_tokens_counted = imported_and_prepared_test.trunc_lems
        test_nb_multitokens = imported_and_prepared_test.nb_multitokens
    
    if params['include_lemma']:
        if params['dev_directory']:
            imported_and_prepared_dev.index_lemmas()
            dev_lemmas_counted = imported_and_prepared_dev.lemmas_unique
            dev_multilabels_lemmas_counted = imported_and_prepared_dev.count_multilabel_lemma()
        if params['test_directory']:
            imported_and_prepared_test.index_lemmas()
            test_lemmas_counted = imported_and_prepared_test.lemmas_unique
            test_multilabels_lemmas_counted = imported_and_prepared_test.count_multilabel_lemma()
    
    if params['dev_directory']:
    #tokens stat for development
        tokens_in_common_dev = set(dev_tokens_counted).intersection(train_tokens_counted)
        dev_tokens_unique = [lemma for lemma in dev_tokens_counted if lemma not in tokens_in_common_dev]
    #tokens stat for test
    else:
        dev_tokens_unique = None
    if params['test_directory']:
        tokens_in_common_test = set(test_tokens_counted).intersection(train_tokens_counted)
        test_tokens_unique = [lemma for lemma in test_tokens_counted if lemma not in tokens_in_common_test]
    else:
        test_tokens_unique = None
    
    
    
    #lemmas stat
    if params['include_lemma']:
        if params['dev_directory']:
            lemmas_in_common_dev = set(dev_lemmas_counted).intersection(train_lemmas_counted)
            dev_lemmas_unique = [lemma for lemma in dev_lemmas_counted if lemma not in lemmas_in_common_dev]
            multilabels_lemmas_in_common_dev = set(dev_multilabels_lemmas_counted).intersection(train_multilabels_lemmas_counted)
            multilabels_lemmas_unique_dev = [lemma for lemma in dev_multilabels_lemmas_counted if lemma not in multilabels_lemmas_in_common_dev]
        if params['test_directory']: 
            lemmas_in_common_test = set(test_lemmas_counted).intersection(train_lemmas_counted)
            test_lemmas_unique = [lemma for lemma in test_lemmas_counted if lemma not in lemmas_in_common_test]
            multilabels_lemmas_in_common_test = set(test_multilabels_lemmas_counted).intersection(train_multilabels_lemmas_counted)
            multilabels_lemmas_unique_test = [lemma for lemma in test_multilabels_lemmas_counted if lemma not in multilabels_lemmas_in_common_test]
    
    
    
    #else:
    #    lemmas_in_common, dev_lemmas_unique, multilabels_lemmas_in_common, multilabels_lemmas_unique = None,None,None,None
    
    corpus_stats_path = os.path.join(params['cellarium_folder'],'Corpus_stats')
    if not os.path.exists(corpus_stats_path):
        os.makedirs(corpus_stats_path)
    
    
    with open(os.path.join(corpus_stats_path, 'corpus_stats.txt'), 'w') as f:
        tokeninfo = 'The training data set has {} tokens in general.'.format(imported_and_prepared_corpus.length_of_corpus)
        f.write(tokeninfo)
        f.write('\n')
        tokeninfo = 'The training data set has {} unique tokens.'.format(len(train_tokens_counted))
        f.write(tokeninfo)
        f.write('\n')
        tokeninfo = 'The training data set has {} multi-tokens.'.format(train_nb_multitokens)
        f.write(tokeninfo)
        f.write('\n')
        if params['dev_directory']:
            tokeninfo = 'The development data set has {} tokens in general.'.format(imported_and_prepared_dev.length_of_corpus)
            f.write(tokeninfo)
            f.write('\n')
            tokeninfo = 'Training and development data sets have {} tokens in common.'.format(len(tokens_in_common_dev))
            f.write(tokeninfo)
            f.write('\n')
            tokeninfo = 'The development data set has {} unique tokens.'.format(len(dev_tokens_unique))
            f.write(tokeninfo)
            f.write('\n')
            tokeninfo = 'The development data set has {} multi-tokens.'.format(dev_nb_multitokens)
            f.write(tokeninfo)
            f.write('\n')
        if params['test_directory']:
            tokeninfo = 'The development test set has {} tokens in general.'.format(imported_and_prepared_test.length_of_corpus)
            f.write(tokeninfo)
            f.write('\n')
            tokeninfo = 'Training and test data sets have {} tokens in common.'.format(len(tokens_in_common_test))
            f.write(tokeninfo)
            f.write('\n')
            tokeninfo = 'The test data set has {} unique tokens.'.format(len(dev_tokens_unique))
            f.write(tokeninfo)
            f.write('\n')
            tokeninfo = 'The test data set has {} multi-tokens.'.format(test_nb_multitokens)
            f.write(tokeninfo)
            f.write('\n')
        
        
        if params['include_lemma']:
            if params['dev_directory']:
                tokeninfo = 'Training and development data sets have {} lemmas in common.'.format(len(lemmas_in_common_dev))
                f.write(tokeninfo)
                f.write('\n')
                tokeninfo = 'A development data set has {} unique lemmas.'.format(len(dev_lemmas_unique))
                f.write(tokeninfo)
                f.write('\n')
                tokeninfo = 'Training and development data sets have {} multilabel lemmas in common.'.format(len(multilabels_lemmas_in_common_dev))
                f.write(tokeninfo)
                f.write('\n')
                tokeninfo = 'A development data set has {} unique multilabel lemmas.'.format(len(multilabels_lemmas_unique_dev))
                f.write(tokeninfo)
                f.write('\n')
            if params['test_directory']:
                tokeninfo = 'Training and test data sets have {} lemmas in common.'.format(len(lemmas_in_common_test))
                f.write(tokeninfo)
                f.write('\n')
                tokeninfo = 'A test data set has {} unique lemmas.'.format(len(test_lemmas_unique))
                f.write(tokeninfo)
                f.write('\n')
                tokeninfo = 'Training and test data sets have {} multilabel lemmas in common.'.format(len(multilabels_lemmas_in_common_test))
                f.write(tokeninfo)
                f.write('\n')
                tokeninfo = 'A test data set has {} unique multilabel lemmas.'.format(len(multilabels_lemmas_unique_test))
                f.write(tokeninfo)
                f.write('\n')
    
    
    
    
        f.close()
    
    
    if params['dev_directory']:
        save_list(os.path.join(corpus_stats_path, 'dev unique tokens.txt'), dev_tokens_unique)
    if params['test_directory']:    
        save_list(os.path.join(corpus_stats_path, 'test unique tokens.txt'), test_tokens_unique)
    
    if params['dev_directory'] and params['include_lemma']:
        save_list(os.path.join(corpus_stats_path, 'dev_lemmas_unique.txt'), dev_lemmas_unique)
        save_list(os.path.join(corpus_stats_path, 'multilabels_lemmas_unique_dev.txt'), multilabels_lemmas_unique_dev)
    
    if params['test_directory']and params['include_lemma']:
        save_list(os.path.join(corpus_stats_path, 'test_lemmas_unique.txt'), test_lemmas_unique)
        save_list(os.path.join(corpus_stats_path, 'multilabels_lemmas_unique_test.txt'), multilabels_lemmas_unique_test)
    
    
    del imported_and_prepared_corpus, imported_and_prepared_dev, imported_and_prepared_test
    
    for colleague in active_workers:
        if colleague == 0:
            continue
        comm.ssend(char_vector_dict, dest = colleague, tag = 111)
        comm.ssend(char_idx, dest = colleague, tag = 111)
        comm.ssend(number_char, dest = colleague, tag = 111)
        comm.ssend(name_of_train_frag, dest = colleague, tag = 111)
        comm.ssend(name_of_val_frag, dest = colleague, tag = 111)
        comm.ssend(name_of_test_frag, dest = colleague, tag = 111)
        comm.ssend(models_path, dest = colleague, tag = 111)
        comm.ssend(train_stats_path, dest = colleague, tag = 111)
        comm.ssend(corpus_stats_path, dest = colleague, tag = 111)
        
    
    '''other processes are waiting for information from process no.0'''
    
    return char_vector_dict, char_idx, number_char,\
    name_of_train_frag, name_of_val_frag, name_of_test_frag,\
    models_path, train_stats_path, corpus_stats_path
    



'''a function for saving prepared data for training'''
def save_tools(tool, name):
    with open('{}.p'.format(name), 'wb') as f:
        print('fragment file format: {}'.format(name))
        pickle.dump(tool, f)
        f.close()
    return

''' a function for loading data'''
def load_tools(name):
    name = name + '.p'
    with open(name, 'rb') as data:
        tool = pickle.load(data)
        data.close()
        return tool


'''a function for Levenshtein distance mapped into 0<x<=1 range'''
def levenshtein(a,b): #hetland.org/coding/python/levenshtein.py
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
        
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    
    #######
    
    l = max(len(a),len(b))
    
    if l == 0:
        l = 1.0
    else:
        l = l
    
    perc = (max(len(a),len(b))-float(current[n]))/l
    
    return perc

'''a mask constructor for batches with sequences of different length'''

def construct_mask(tensor_size,length):
    
   
    
    blanket = torch.zeros(tensor_size).byte().permute(1,0)
    
    list_of_mask_len = []
    
    for x in range(len(length)):
        ones = [z+1 for z in range(length[x])]
        while len(ones)<tensor_size[0]:
            ones.append(0)
        list_of_mask_len.append(ones)
    
    
   
    mask = torch.LongTensor(list_of_mask_len)
  
    masking = blanket.scatter_(1, mask,1)
 
    return masking

'''a function for detaching hidden Variables from their graph history'''
def repackage_variable(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_variable(v) for v in h)



def loading_text_fragment(text_fragment=None, include_lemma=None, include_pos=None, include_morph=None):
    
    '''
    a text fragment is an output of the codecs open function
    '''
    corpus = {'token': []}
    if include_lemma:
        corpus['lemma'] = []
    if include_pos:
        corpus['pos'] = []
    if include_morph:
        corpus['morph'] = []
    
    
    for line in text_fragment:
            
            if line and not line[0] == '@':
                try:
                    comps = line.split('\t')
                    
                    tok = comps[0].lower()#.strip()
                    #tok = leave_only_alphanumeric(tok)
                    #if self.v2u:
                    #    tok = tok.replace('v', 'u')
                    if include_lemma:
                        lem = comps[1].lower().strip().replace('-', '')
                    if include_pos:
                        pos = comps[2]
                    if include_morph:
                        # morph = '|'.join(sorted(set(comps[3].split('|'))))
                        morph = '|'.join(comps[3].split('|'))
                    tok = tok.strip().replace('~', '')#.replace(' ', '')
                    if tok == '': #for &-like cases
                        tok = ' '
                    
                    corpus['token'].append(tok)
                    if include_lemma:
                        corpus['lemma'].append(lem)
                    if include_pos:
                        corpus['pos'].append(pos)
                    if include_morph:
                        corpus['morph'].append(morph)
                except:
                    print(text_fragment, ':', line)
    
    return corpus


def save_list(file_name, list_to_save, limit=-1):
        
        with open(file_name, 'w') as f:
            for nb, element in enumerate(list_to_save):
                f.write(element.encode('utf-8'))
                f.write('\n')
                if limit == nb:
                    break
            
            f.close()
        
        return

def read_list(file_name):
        #codecs.open(filepath, 'r', encoding='utf8')
        list_to_read = []
        file_to_read = codecs.open(file_name, 'r', encoding='utf8')
        for line in file_to_read:
            list_to_read.append(line.strip())
        file_to_read.close()
        
        return list_to_read


def alternate_iterators(*iters): #https://stackoverflow.com/questions/2017895/alternating-between-iterators-in-python  
    Dummy=object()

    for elt in itertools.chain.from_iterable(
        itertools.izip_longest(*iters,fillvalue=Dummy)):
        if elt is not Dummy:

            yield elt


def generate_dictionary_samples(dictionary=None, lemma=None, pos=None, morph=None, batch_size=3):
    
    
    corpus_loop = range(len(dictionary['token']))
    random.seed(time.time())
    random.shuffle(corpus_loop)
    corpus_loop = [(x,y) for x,y in enumerate(corpus_loop)]
    
    
    
    for nb, z in corpus_loop:
        
        if nb % batch_size == 0:
            tokens, lemmas, poss, morphs = [], [], [], []
        
            for x in range(batch_size):
                
                try:
                    y = corpus_loop[nb + x][1]
                    
                    token = dictionary['token'][y]
                    
                    tokens.append(token)
                    
                    if lemma:
                        lemma_ = dictionary['lemma'][y]
                        lemmas.append(lemma_)
                    if pos:
                        pos_ = dictionary['pos'][y]
                        poss.append(pos_)
                    if morph:
                        morph_ = dictionary['morph'][y]
                        morphs.append(morph_)
                
                except:
                    continue
                
            yield tokens, lemmas, poss, morphs, None, None
    

      






'''generator for corpus elements'''

def generate_text_samples(corpus=None, lemma=None, pos=None, morph=None, context_size=10, batch_size=3):
    
    corpus_loop = range(len(corpus['token']))
    random.seed(time.time())
    random.shuffle(corpus_loop)
    corpus_loop = [(x,y) for x,y in enumerate(corpus_loop)]
    
    for nb, z in corpus_loop:
        
        if nb % batch_size == 0:
            tokens, lemmas, poss, morphs, context_left_list, context_right_list = [], [], [], [], [], []
        
            for x in range(batch_size):
                
                try:
                    y = corpus_loop[nb + x][1]
                    
                    
                    token = corpus['token'][y]
                    
                    
                    
                    tokens.append(token)
                    
                    if lemma:
                        lemma_ = corpus['lemma'][y]
                        lemmas.append(lemma_)
                    if pos:
                        pos_ = corpus['pos'][y]
                        poss.append(pos_)
                    if morph:
                        morph_ = corpus['morph'][y]
                        morphs.append(morph_)
                        
                    
                    
                    #context
                    
                
                    if (y-context_size) < 0:
                        context_left_1 = corpus['token'][(y-context_size):]
                        context_left_2 = corpus['token'][:y]
                        context_left = context_left_1 +context_left_2
                        
                    else:
                        context_left = corpus['token'][(y-context_size):y]
                    
                    context_left_list.append(context_left)
                    
                    
                        
                    
                    if y+context_size <= len(corpus['token']):
                        context_right = corpus['token'][y:y+context_size]
                        
                    else:
                        context_right_1 = corpus['token'][y:]
                        context_right_2 = corpus['token'][:context_size-(len(corpus['token'])-y)]
                        context_right = context_right_1 + context_right_2
                        
                    context_right_list.append(context_right)
                
                except:
                    continue
                
            yield tokens, lemmas, poss, morphs, context_left_list, context_right_list 
    
 
def generate_corpus_samples(corpus=None, lemma=None, pos=None, morph=None, dictionary=None,context_size=10, batch_size=3):
    
    if dictionary:
    
        return alternate_iterators(
            generate_text_samples(corpus, lemma, pos, 
                              morph, context_size, batch_size),
            generate_dictionary_samples(dictionary, lemma, pos, morph, batch_size)
                    
            )

    else:
        
        return generate_text_samples(corpus, lemma, pos, 
                              morph, context_size, batch_size)
        

'''a function for creating sequences with predefined lemma, POS and/or morphological data'''
def create_sequences(instances, min_seq_len, max_seq_len,include_lemma, include_pos, include_morph):
    
      
    list_of_material = instances['token'] 
    material_length = len(list_of_material)
 
    
    
    start = random.randint(0,max_seq_len)
    
    recent_end = 0
    
    last_end = 0
    
    
    list_of_seq = []
    
    
    while recent_end < material_length:
        sequence_data = {}
        start = last_end 
        
        
        recent_end = start + random.randint(min_seq_len,max_seq_len)
     
        sequence = instances['token'][start:recent_end]
        sequence_data['token'] = sequence
        
        if include_lemma:
            sequence = instances['lemma'][start:recent_end]
            seq = []
            for element in sequence:
                element = element.split('+')
                seq.extend(element)
            sequence_data['lemma'] = seq
            
        if include_pos:
            sequence = instances['pos'][start:recent_end]
            seq = []
            for element in sequence:
                element = element.split('+')
                seq.extend(element)
            sequence_data['pos'] = seq
            
        if include_morph:
            sequence = instances['morph'][start:recent_end]
            seq = []
            for element in sequence:
                element = element.split('+')
                seq.extend(element)
            sequence_data['morph'] = seq
            
        last_end = recent_end#-1#for overlap
        list_of_seq.append(sequence_data)
        
    return list_of_seq


'''the first prototype function for vectorising tokens'''

def vectorize_token1(token=None, char_vector_dict=None, max_len_tok=None):

  
    token = token[:max_len_tok]
    
   
    filler = np.zeros(len(char_vector_dict), dtype='float32')

    tok_X = []
    for char in token:
        try:
            tok_X.append(char_vector_dict[char])
        except KeyError:
            tok_X.append(filler)
    
    while len(tok_X) < max_len_tok:
        tok_X.append(filler)
    
    numpy_matrix = np.array(tok_X, dtype='float32')
    
   
    
    return torch.from_numpy(numpy_matrix)



'''the second prototype function for vectorising tokens'''


def vectorize_token2(token=None, encoder_letters=None,
                     max_len_tok=None):

   
    token = token[:max_len_tok]
    
    
    #label_encoder = encoder_letters
    #label_encoder = one_hot_encoder
    
    tok_X = []
    for char in token:
        try:
            let_indexed = encoder_letters.transform([char])
            
            let_indexed = torch.from_numpy(let_indexed).long()
            
            tok_X.append(let_indexed)
        except:
            let_indexed = encoder_letters.transform(['<UNK>'])
            
            let_indexed = torch.from_numpy(let_indexed).long()
            tok_X.append(let_indexed)
        
         
       
    filler = encoder_letters.transform(['<PAD>'])
    filler = torch.from_numpy(filler).long()
    
   
    while len(tok_X) < max_len_tok:
        tok_X.append(filler)
    
    tok_X = torch.cat(tok_X,0)
    tok_X = torch.unsqueeze(tok_X, 0)
    tok_X = torch.unsqueeze(tok_X, 0)#twice!!!!
    
    #print(tok_X)
    #quit()
    
    return tok_X

'''a function for input construction'''


def construct_input_repr(sequence=None,
                            encoder_letters=None,
                            max_len_tok=None,
                            context= False
                            ):
    
    #label_encoder, one_hot_encoder = encoder_letters
    
    
    list_of_matrices = []
    
    #PAD = label_encoder.transform(['<PAD>'])
    
    #PAD = torch.from_numpy(PAD).long()
    #PAD = torch.unsqueeze(PAD, 0)
    #PAD = PAD.expand(1,max_len_tok)
    
    
    #SOS = label_encoder.transform(['<SOS>'])
    
    #SOS = torch.from_numpy(SOS).long()
    
    
    #SOS = torch.unsqueeze(SOS, 0)
    #SOS = SOS.expand(1,max_len_tok)
    
    
    
    #EOS = label_encoder.transform(['<EOS>'])
    
    #EOS = torch.from_numpy(EOS).long()
    
    
    #EOS = torch.unsqueeze(EOS, 0)
    #EOS = EOS.expand(1,max_len_tok)
    
    for tokens in sequence:
        
        list_ = []
        
        if not context:
            
            
            
            token_ = tokens.split('+')
            for tok in token_:
                token_vectorised = vectorize_token2(token=tok,
                                           encoder_letters=encoder_letters,
                                           max_len_tok=max_len_tok)
                
                list_.append(token_vectorised.contiguous().permute(2,1,0).contiguous())
            
            
            list_ = torch.cat(list_,0)

            list_of_matrices.append(list_)
        
        if context:
            
            
            for tok in tokens:
                token_vectorised = vectorize_token2(token=tok,
                                               encoder_letters=encoder_letters,
                                               max_len_tok=max_len_tok)
                    
               
                list_.append(token_vectorised)
            
            
            list_ = torch.cat(list_,0)
            
            list_of_matrices.append(list_)  
              
       
     
     
    
      
    
    
    torchy_seq = torch.cat(list_of_matrices,1)
    
    return torchy_seq


'''the first prototype function for constructing output data'''

def construct_output_repr1(sequence=None, encoder=None, max_len_lemma=15):
    
   
    list_of_elements = []  
    
    PAD = encoder.transform(['<PAD>']).tolist()
    PLUS = encoder.transform(['+']).tolist()
    
    for element in sequence:
        element_repr = []
       
        #element = element.split('+')
        
        
        for num in range(len(element)):#range(3):
            try:
                chars = []
                for char in list(element[num]):
                    
                    try:
                        char = encoder.transform([char]).tolist()
                    except:
                        char = encoder.transform(['<UNK>']).tolist()
                    chars.extend(char)
               
                
                
                element_repr.extend(chars)
                
            except Exception as e:
                
                print(e)
                print(element_repr)
                quit()
                pass
                #empty = encoder.transform(['<PAD>']).tolist()
                #element_repr.extend(empty*max_len_lemma)
        #print(element_repr)
        while len(element_repr) < max_len_lemma:
            element_repr.extend(PAD)
            
        list_of_elements.append(element_repr[:max_len_lemma])
    
    #for l in list_of_elements:
    #    print(len(l))
    #quit()
    list_of_elements = np.asarray(list_of_elements, dtype=float)
    
    torchy_tensor = torch.from_numpy(list_of_elements).long()
    #print(torchy_tensor.size())
    #quit()
    
    return torchy_tensor

'''the second prototype function for constructing output data'''

def construct_output_repr2(sequence=None, encoder=None, max_tag_len_seq=3):
    list_of_elements = []  
    
    PAD = encoder.transform(['<PAD>'])
    
    
    for element in sequence:
        
        element = element.replace('+', ' ').replace('>', ' > ').replace('<', ' < ').split()
        #print(element)
        enhanced_element = []
        for elem in element:
            try:
               
                    
                try:
                    char = encoder.transform([elem]).tolist()
                except:
                    char = encoder.transform(['<UNK>']).tolist()
                    
                enhanced_element.extend(char)
                
            except Exception as e:
                
                print(e)
                print(char)
                quit()
                pass
                #empty = encoder.transform(['<PAD>']).tolist()
                #element_repr.extend(empty*max_len_lemma)
        #print(element_repr)
        while len(enhanced_element) < max_tag_len_seq:
            enhanced_element.extend(PAD)
       
        list_of_elements.append(enhanced_element[:max_tag_len_seq])
    #list_of_elements.extend(EOS)
    
    #print(list_of_elements)
    
    
    torchy_tensor = torch.LongTensor(list_of_elements).view(-1)
  
    return torchy_tensor

def inverse_transform(result_list, encoder_lemma):
    for batch_frag in result_list: #single out a batch fragment
        #:#for position in batch_frag: #single out a position in batch fragment
        lemma_reconstructed = encoder_lemma.inverse_transform(batch_frag)
            #lemma_reconstructed = [lemma for lemma in lemma_reconstructed if lemma !='<PAD>']
            #lemma_reconstructed = [lemma for lemma in lemma_reconstructed if lemma !='<UNK>']
        yield ''.join(lemma_reconstructed)

def inverse_transform_pos_morph(result_list, original_list,encoder):
    results, originals = [], []
    overlap = 0.0
    nb_originals = 0.0
    multi_token_original = 0.0
    multi_token_result = 0.0
    multi_tag_original = 0.0
    multi_tag_result = 0.0
    for nb, batch_frag in enumerate(result_list): #single out a batch fragment
        #for position in batch_frag: #single out a position in batch fragment
        result = encoder.inverse_transform(batch_frag).tolist()
        
        original = encoder.inverse_transform(original_list[nb]).tolist()
        #print(original)
        overlap += len([i for i, j in zip(result, original) if i == j and i != '<PAD>'])
        #quit()
        result = [x for x in result if x !='<PAD>']
        original = [x for x in original if x !='<PAD>']
        if '<' in original or '>' in original:
            multi_token_original += len(original)
            if ('<' in result or '>' in result):# and ('<' in original or '>' in original):
                multi_token_result += len(result)
        if '+' in original:
            multi_tag_original += len(original)
            if '+' in result:
                multi_tag_result += len(result)
        
        
        nb_originals += len(original)
        results.append(' '.join(result))
        originals.append(' '.join(original))
    #if multi_tag_original == 0.0:
    #    multi_tag_original = 1.0
    #if multi_token_original == 0.0:
    #    multi_token_original = 1.0
    
        
    return overlap/nb_originals, results, originals, multi_token_result, multi_token_original, multi_tag_result, multi_tag_original
 





def find_multitoken_tags(tag_list):#lemma,tag,morph
    length = len(tag_list)
    new_list = []
    for nb, tag in enumerate(tag_list):
        if '<' in tag or '>' in tag:
            new_list.append(nb)
        if '<' in tag and '>' in tag:
            new_list.append(nb)
    
    return new_list

def find_unique_elements(token_list, unique_list):#lemma,tag,morph
    length = len(token_list)
    new_list = []
    for nb, tag in enumerate(token_list):
        if tag in unique_list:
            new_list.append(nb)
    
    return new_list

def find_multitag_tokens(tag_list):#lemma,tag,morph
    length = len(tag_list)
    new_list = []
    for nb, tag in enumerate(tag_list):
        if '+' in tag:
            new_list.append(nb)
    
    return new_list

                
def check_lemma_list(lemma_list, results):
    results_rechecked=[]
    for r in results:
        scores = []
        for l in lemma_list:
            scores.append(fuzz.ratio(r,l))
        max_score = max(scores)
        results_rechecked.append(lemma_list[scores.index(max_score)])
    
    return results_rechecked    
        
def count_lemmas_validation(result_list, lemma_list, 
                            lemmas_multitoken_idx, lemmas_multitag_idx, 
                            tokens_unique_idx, lemmas_unique_idx, multilabels_lemmas_unique_idx
                            ):
    
    r = 0.0
    tokens_unique = 0.0
    multi_token = 0.0
    multi_tag = 0.0
    tag_unique = 0.0
    multi_tag_unique = 0.0
    for nb, element in enumerate(result_list):
        
        r += 1.0 if element == lemma_list[nb] else 0.0 #fuzz.token_sort_ratio(element,lemma_list[nb])/100.0
        tokens_unique += 1.0 if element == lemma_list[nb] and nb in tokens_unique_idx else 0.0
        multi_token += 1.0 if element == lemma_list[nb] and nb in lemmas_multitoken_idx else 0.0
        multi_tag += 1.0 if element == lemma_list[nb] and nb in lemmas_multitag_idx else 0.0
        tag_unique += 1.0 if element == lemma_list[nb] and nb in lemmas_unique_idx else 0.0
        multi_tag_unique += 1.0 if element == lemma_list[nb] and nb in multilabels_lemmas_unique_idx else 0.0
    
    if len(tokens_unique_idx) == 0:
        len_tokens_unique_idx=1
    else:
        len_tokens_unique_idx=len(tokens_unique_idx)    
        
    if len(lemmas_multitoken_idx) == 0:
        len_lemmas_multitoken_idx=1
    else:
        len_lemmas_multitoken_idx=len(lemmas_multitoken_idx)
    
    if len(lemmas_multitag_idx) == 0:
        len_lemmas_multitag_idx=1
    else:
        len_lemmas_multitag_idx=len(lemmas_multitag_idx)    
    
    if len(lemmas_unique_idx) == 0:
        len_lemmas_unique_idx=1
    else:
        len_lemmas_unique_idx=len(lemmas_unique_idx)    

    if len(multilabels_lemmas_unique_idx) == 0:
        len_multilabels_lemmas_unique_idx=1
    else:
        len_multilabels_lemmas_unique_idx=len(multilabels_lemmas_unique_idx)    
    
    
    return r/(nb+1), multi_token/len_lemmas_multitoken_idx, multi_tag/len_lemmas_multitag_idx, \
        tokens_unique/len_tokens_unique_idx, tag_unique/len_lemmas_unique_idx, multi_tag_unique/len_multilabels_lemmas_unique_idx


'''a function for drawing a plot for training statistics'''
    
def save_plot(points, target_folder=None, plot_name=None, labels=None,base=1):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=base)
    ax.yaxis.set_major_locator(loc)
    #print(len(points))
    plt.plot(points)
    if labels:
        plt.legend(labels)
    plt.savefig(os.path.join(target_folder, plot_name))
    plt.close('all')
    #fig.clf()
    del loc, fig, ax
    gc.collect()
    
    return
        


'''a function for training statistics'''

#class training_statistics():

def training_statistics(
        loss_lemmatizer = None,
        loss_pos = None,
        loss_morph = None,
        train_and_val = None,
        train_stats_folder = None,
        lr_mod_param = 1.0,
        lr_mod_moment = 1,
        number_lr_mods = 1,
        current_iter=1,
        n_iters=1,
        n_val=1,
        rank = None,
        active_workers = None,
        start_time = None,
        ):
    
    '''define names for plot files and for a file with training and validation statistic data'''
    
    file_name_for_train_loss = "train_loss_proc_{}.p".format(rank)
    
    train_plot_lemmatizer = "train_loss_plot_lemmatizer_{}.png".format(rank)
    train_plot_pos_tagger = "train_loss_plot_pos_tagger_{}.png".format(rank)
    train_plot_morph_tagger = "train_loss_plot_morph_tagger_{}.png".format(rank)
    
    '''define plot names for validation statistics'''
    
    file_name_for_val_loss = "val_loss_proc_{}.p".format(rank)
    
    
    '''accuracy plot names'''
    
    val_lemmatizer_plot_name1 = "val_results_lemmatizer_plot_proc_{}.png".format(rank)
    val_pos_tagging_plot_name1 = "val_results_pos_tagging_plot_proc_{}.png".format(rank)
    val_morph_tagging_plot_name1 = "val_results_morph_tagging_plot_proc_{}.png".format(rank)
    
    '''loss plot names'''
    
    val_lemmatizer_plot_name2 = "val_loss_lemmatizer_plot_proc_{}.png".format(rank)
    val_pos_tagging_plot_name2 = "val_loss_pos_tagging_plot_proc_{}.png".format(rank)
    val_morph_tagging_plot_name2 = "val_loss_morph_tagging_plot_proc_{}.png".format(rank)


    
    '''load or generate a file with statistics'''
    
    try:
        file_with_pickle_loss_info = open(os.path.join(train_stats_folder, file_name_for_train_loss), "rb")
        loss_general = pickle.load(file_with_pickle_loss_info)
        file_with_pickle_loss_info.close()
        print('loaded loss data...')
    except:
        loss_general = [[],[],[]]
    
    
    '''add training results to statistics'''
        
    if loss_lemmatizer:
        loss_general[0].extend([loss_lemmatizer])
    if loss_pos:
        loss_general[1].extend([loss_pos])
    if loss_morph:
        loss_general[2].extend([loss_morph])
    
    '''save the training statistics'''
    file_to_pickle = open(os.path.join(train_stats_folder,file_name_for_train_loss),"wb")
    pickle.dump(loss_general, file_to_pickle)
    file_to_pickle.close()
    '''draw new stat plots for training'''
    
    if loss_lemmatizer:
        base_lemma = round(max(loss_general[0])/10,5)
        save_plot(loss_general[0], target_folder=train_stats_folder, plot_name=train_plot_lemmatizer, base = base_lemma)
    if loss_pos:
        base_pos = round(max(loss_general[1])/10,5)
        save_plot(loss_general[1], target_folder=train_stats_folder, plot_name=train_plot_pos_tagger, base = base_pos)
    if loss_morph:
        base_morph = round(max(loss_general[2])/10,5)
        save_plot(loss_general[2], target_folder=train_stats_folder, plot_name=train_plot_morph_tagger, base = base_morph)
    
    
    '''adopt a new learning rate depending on the number of epochs so far'''
    
    if current_iter % lr_mod_moment == 1 and current_iter != 0:
        number_lr_mods += 1
        
        lr_mod_param = 1.0
        
        lr_remod_param = 1/lr_mod_param
        
        
        for param_group in train_and_val.model_optimizer.param_groups:
            #if train_and_val.exchange.update_done == True:
            #    lr = param_group['lr']
            #    lr = round((lr_remod_param**number_lr_mods)*lr,15)
            #    param_group['lr'] = lr
            #    number_lr_mods = 0
            #else:
            lr = param_group['lr']
            lr = round(lr_mod_param*lr,15)
            param_group['lr'] = lr
    
      
    '''perform routine validation during training'''
    
    if current_iter % n_val == 0:
        
        
        '''load or generate a file with validation statistics'''
        
        try:
            file_with_pickle_val_loss = open(os.path.join(train_stats_folder, file_name_for_val_loss), "rb")
            loss_general_val = pickle.load(file_with_pickle_val_loss)
            file_with_pickle_val_loss.close()
            print('loaded val loss data...')
        except:
            empty_list_x1 = []
            empty_list_x3 =[[],[],[]]
            empty_list_x6 =[[],[],[],[],[],[]]
            loss_general_val = [list(empty_list_x6),list(empty_list_x3),list(empty_list_x3),list(empty_list_x1),list(empty_list_x1),list(empty_list_x1)]
  
        results_lemma, results_pos, results_morph, loss_lemma, loss_pos, loss_morph = train_and_val.valid_dev(nth_iter = current_iter, development=True)
        
        #print(results_lemma, results_pos, results_morph, loss_lemma, loss_pos, loss_morph,'gdfgdgfd')
        #quit()
        '''add new validation statistics'''
        
        if results_lemma != None:
            loss_general_val[0] = [elem1+[elem2] for nb1, elem1 in enumerate(loss_general_val[0]) for nb2, elem2 in enumerate(results_lemma) if nb1 == nb2]
        if results_pos != None:
            loss_general_val[1] = [elem1+[elem2] for nb1, elem1 in enumerate(loss_general_val[1]) for nb2, elem2 in enumerate(results_pos) if nb1 == nb2]
        if results_morph != None:
            loss_general_val[2] = [elem1+[elem2] for nb1, elem1 in enumerate(loss_general_val[2]) for nb2, elem2 in enumerate(results_morph) if nb1 == nb2]
        
        if loss_lemma != None:
            loss_general_val[3].extend([loss_lemma])
        if loss_pos != None:
            loss_general_val[4].extend([loss_pos])
        if loss_morph != None:
            loss_general_val[5].extend([loss_morph])
        
        '''save validation statistics'''
        new_file_with_pickle_loss_info = open(os.path.join(train_stats_folder,file_name_for_val_loss), "wb")
        pickle.dump(loss_general_val, new_file_with_pickle_loss_info)
        new_file_with_pickle_loss_info.close()
        
        '''draw plats for validation statistics'''
        
      
        labels = ('general', 'multilabel', 'multitoken', 'unique_tokens', 'unique_lemmas', 'unique_multilabel_lemmas')
        
        if results_lemma != None:
            gen, multitag, multitoken, unique_t, unique_l, unique_multi_l = loss_general_val[0]
            base_ = round(max(gen)/10,5)
            base_ = 0.1#base_ if base_ > 0 and len(gen) != 1 and round(max(gen)-min(gen),2) != 0.0 else 1
            for_plot =list(zip(gen, multitag, multitoken,unique_t, unique_l, unique_multi_l))
            save_plot(for_plot, target_folder=train_stats_folder, plot_name=val_lemmatizer_plot_name1, labels=labels, base = base_)
        if results_pos != None:
            gen, multitag, multitoken = loss_general_val[1]
            base_ = round(max(gen)/10,5)
            base_ = 0.1# base_ if base_ > 0 and len(gen) != 1 and round(max(gen)-min(gen),2) != 0.0 else 1
            #print(base_, gen)
            for_plot =list(zip(gen, multitag, multitoken))
            save_plot(for_plot, target_folder=train_stats_folder, plot_name=val_pos_tagging_plot_name1, labels=labels, base = base_)
        if results_morph != None:
            gen, multitag, multitoken = loss_general_val[2]
            base_ = round(max(gen)/10,5)
            base_ = 0.1#base_ if base_ > 0 and len(gen) != 1 and round(max(gen)-min(gen),2) != 0.0 else 1
            for_plot =list(zip(gen, multitag, multitoken))
            save_plot(for_plot, target_folder=train_stats_folder, plot_name=val_morph_tagging_plot_name1, labels=labels, base = base_)
        
        if loss_lemma != None:
            base_ = round(max(loss_general_val[3])/10,5)
            save_plot(loss_general_val[3], target_folder=train_stats_folder, plot_name=val_lemmatizer_plot_name2, base = base_)
        if loss_pos != None:
            base_ = round(max(loss_general_val[4])/10,5)
            save_plot(loss_general_val[4], target_folder=train_stats_folder, plot_name=val_pos_tagging_plot_name2, base = base_)
        if loss_morph != None:
            base_ = round(max(loss_general_val[5])/10,5)
            save_plot(loss_general_val[5], target_folder=train_stats_folder, plot_name=val_morph_tagging_plot_name2, base = base_)
        
    gc.collect()
    
    return

'''a function for training with build-in validation'''
#@profile
def train_iteretions(
      train_and_val=None,
      train_stats_folder=None,
      n_iters=1,
      lr_mod_param=1.0,
      lr_mod_moment = 1,
      n_val=1,
      rank = None,
      active_workers = None,
      comm = None,
      use_cuda = None,
      start_time = None
      ):
    
    
    '''a variable for tracking lr modifications'''
    
    number_lr_mods = 0
    
    '''start training for 'n_iters' iterations'''
    for i in range(n_iters): 
        
        loss_lemmatizer, loss_pos, loss_morph \
        = train_and_val.train(nth_iter=i, start_time=start_time)
        
        training_statistics(loss_lemmatizer=loss_lemmatizer, 
                            loss_pos=loss_pos, loss_morph=loss_morph,
                            train_and_val = train_and_val,
                            train_stats_folder = train_stats_folder,
                            lr_mod_param = lr_mod_param,
                            lr_mod_moment = lr_mod_moment,
                            number_lr_mods = number_lr_mods,
                            current_iter=i,
                            n_iters=n_iters,
                            n_val=n_val,
                            rank = rank,
                            active_workers = active_workers
                           )
          
    '''waiting for other processes and exchange of last pieces of information'''
    
    train_and_val.catch_up()
        
        
    '''calling final messages after training and averaging the model'''
    
    if rank != 0:
        
        msg_weights = []
        for param in train_and_val.model.parameters():
            msg_weight= torch.div(param.data.cpu(), len(active_workers))
            msg_weights.append(msg_weight)
        
        
        msg = ("{}".format(rank), msg_weights)
        comm.ssend(msg, dest=0, tag=1)
        print("finished training on {}".format(rank))
    else:
        for param in train_and_val.model.parameters():
            param.data = torch.div(param.data, len(active_workers))
        
        for x in range(len(active_workers) - 1):
            msg, weights = comm.recv(tag=1)
            print('averaging weights from process no.{}'.format(msg))
            for x_id, param in enumerate(train_and_val.model.parameters()):
                        
                        if use_cuda:
                            weight = weights[x_id].cuda()
                            param.data = param.data + weight
                        else:
                            
                            param.data = param.data + weights[x_id]
            
        print("finished training and model averaging")
    
    
    
        '''final validation'''
        #if rank == 0:results_lemma_multitag, results_lemma_multitoken
        
        
        results_lemma, results_pos, results_morph, loss_lemma, loss_pos, loss_morph \
        = train_and_val.valid_dev()
        
        if results_lemma != None:
            accuracy1 = 'accuracy for lemmatizer after {} epochs: {}% all lemmas, {}% multitags, {}% multitokens, {}% lemmas for unique tokens, {}% unique lemmas, {}% unique multilabel tokens'.format(n_iters, \
            (results_lemma[0]*100),(results_lemma[1]*100),(results_lemma[2]*100),\
            (results_lemma[3]*100),(results_lemma[4]*100),(results_lemma[5]*100))
            print(accuracy1)
        if results_pos != None:
            accuracy2 = 'accuracy for pos tagging after {} epochs: {}% all poss, {}% multitags, {}% multitokens'.format(n_iters, (results_pos[0]*100),(results_pos[1]*100),(results_pos[2]*100))
            print(accuracy2)
        if results_morph != None:
            accuracy3 = 'accuracy for morph tagging after {} epochs: {}% all poss, {}% multitags, {}% multitokens'.format(n_iters, (results_morph[0]*100),(results_morph[1]*100),(results_morph[2]*100))
            print(accuracy3)
            
        with open(os.path.join(train_stats_folder, "info_from_training.txt"), "a") as myfile:
                    
            myfile.write('\n')
            if results_lemma != None:
                myfile.write(accuracy1)
                myfile.write('\n')
            if results_pos != None:
                myfile.write(accuracy2)
                myfile.write('\n') 
            if results_morph != None:
                myfile.write(accuracy3)
                myfile.write('\n')
   
    
def generator_for_tagging(text_in_list=None,
                          batch_size = None,
                          context_size = None
                          ):
    
    for z in range(len(text_in_list)):
        
        if z % batch_size == 0:
            tokens, context_left_list, context_right_list = [], [], []
            
            for x in range(batch_size):  
                y = z + x    
                try:
                    token = text_in_list[y]
                    tokens.append(token)
                except:
                    continue
                
                if (y-context_size) < 0:
                    context_left_1 = text_in_list[(y-context_size):]
                    context_left_2 = text_in_list[:y]
                    context_left = context_left_1 +context_left_2
                    
                else:
                    context_left = text_in_list[(y-context_size):y]
                
                context_left_list.append(context_left)
                
                
                    
                
                if y+context_size <= len(text_in_list):
                    context_right = text_in_list[y:y+context_size]
                    
                else:
                    context_right_1 = text_in_list[y:]
                    context_right_2 = text_in_list[:context_size-(len(text_in_list)-y)]
                    context_right = context_right_1 + context_right_2
                    
                context_right_list.append(context_right)
                
            yield tokens, context_left_list, context_right_list   
                
            

def tagging(
        params=None,
        use_cuda=False
        ):
    
    folder1 = params['texts_to_tag_directory']
    folder2 = params['texts_tagged_directory']
    
    if not os.path.exists(folder2):
        os.makedirs(params['texts_tagged_directory'])
    
    
    for text in os.listdir(folder1):
        text_in_list = []
        text_tagged = {'tokens': []}
        if params['include_lemma']:
            text_tagged['lemma'] = []
        if params['include_pos']:
            text_tagged['pos'] = []
        if params['include_morph']:
            text_tagged['morph'] = []
        
        t = codecs.open(os.path.join(folder1,text),encoding='utf8')
        
        word_line_coordinates = []
        for nb_line,line in enumerate(t):
            #print(line.encode('utf-8'))
            #quit()
            for nb_word, word in enumerate(line.split()):
                word = leave_only_alphanumeric(word)
                word = word.lower().strip()
                #print(word)
                text_in_list.append(word)
                word_line_coordinates.append([nb_line,nb_word])
            
        encoder_lemma = None
        encoder_pos = None
        encoder_morph = None
        
        lemma_size = 1
        pos_size = 1
        morph_size = 1
        
        try:
            encoder_letters = load_tools(os.path.join(params['cellarium_folder'], 'encoder_letters'))
            letters_size = len(encoder_letters[0].classes_)
            if params['include_lemma']:
                encoder_lemma = load_tools(os.path.join(params['cellarium_folder'], 'encoder_lemma'))
                lemma_size = len(encoder_lemma.classes_)
            if params['include_pos']:
                encoder_pos = load_tools(os.path.join(params['cellarium_folder'], 'encoder_pos'))
                pos_size = len(encoder_pos.classes_)
            if params['include_morph']:
                encoder_morph = load_tools(os.path.join(params['cellarium_folder'], 'encoder_morph'))
                morph_size = len(encoder_morph.classes_)
                
        except:
            print('In your cellarium folder there must be encoders')
            quit()
            
        model = ModelConvDecon(embedder_size=params['embedder_letters_hidden_size'],
                          hidden_size_model=params['hidden_size_model'],
                         include_lemma = params['include_lemma'], include_pos = params['include_pos'], include_morph = params['include_morph'],
                         max_len_tok= params['max_len_tok'],
                         letters_size=letters_size,
                         lemma_size= lemma_size, pos_size=pos_size, morph_size=morph_size,                         context_size=params['context_size'],
                         max_len_lemma=params['max_len_lemma'],
                         max_tag_len_seq=params['max_tag_len_seq']
                         )
        if use_cuda:
            model.cuda()
        
        
        try:
            model.load_state_dict(torch.load(os.path.join(params['cellarium_folder'], "model_saved_proc_{}".format(0))))
            print('Loaded trained model')
        except:
            print('You must have a trained model.')
            quit()
        
        
        
        
        
        for x in generator_for_tagging(text_in_list, params['batch_size'], params['context_size']):
            
            tokens, context_left_list, context_right_list = x
            
            #print(tokens)
            text_tagged['tokens'].extend(tokens)
            
            
            tokens_vectorised = construct_input_repr(tokens, encoder_letters, params['max_len_tok'], context=False)
            L_context_vectorised = construct_input_repr(context_left_list, encoder_letters, params['max_len_tok'], context=True)
            R_context_vectorised = construct_input_repr(context_right_list, encoder_letters, params['max_len_tok'], context=True)
            
            output_lemmatizer, output_tagger_pos, output_tagger_morph = \
            model(tokens=tokens_vectorised,
                context_left=L_context_vectorised, 
                context_right=R_context_vectorised
                )
            
            if params['include_lemma']:
                output_lemmatizer = output_lemmatizer.view(-1,params['max_len_lemma'],lemma_size).data.cpu().topk(1,dim=2)[1].contiguous().view(-1,params['max_len_lemma']).numpy().tolist()
                output_lemmatizer = [result.replace('<PAD>','') for result in inverse_transform(output_lemmatizer, encoder_lemma)]
                #output_lemmatizer = encoder_lemma.inverse_transform(output_lemmatizer)
                text_tagged['lemma'].extend(output_lemmatizer)
            if params['include_pos']:
                output_tagger_pos = output_tagger_pos.data.cpu().topk(1,dim=1)[1].view(-1,params['max_tag_len_seq']).numpy().tolist()
                output_tagger_pos = [result.replace('<PAD>','') for result in inverse_transform(output_tagger_pos, encoder_pos)]
                #output_tagger_pos = encoder_pos.inverse_transform(output_tagger_pos)
                text_tagged['pos'].extend(output_tagger_pos)
            if params['include_morph']:
                output_tagger_morph = output_tagger_morph.data.cpu().topk(1,dim=1)[1].view(-1,params['max_tag_len_seq']).numpy().tolist()
                output_tagger_pos = [result.replace('<PAD>','') for result in inverse_transform(output_tagger_morph, encoder_morph)]
                #output_tagger_morph = encoder_morph.inverse_transform(output_tagger_morph)
                text_tagged['morph'].extend(output_tagger_morph)

        #
        #print(text_tagged['tokens'])
        #quit()
        line_number = 0
        with open(os.path.join(folder2,text),'wb') as p:
            for number, token in enumerate(text_tagged['tokens']):
                nb_line = word_line_coordinates[number][0]
                if nb_line != line_number:
                    p.write('\n')
                    line_number += 1
                '''since token variable consists of as many tokens as batch size'''
                #for num, tok in enumerate(token): 
                token_plus_description = []
                token_plus_description.append(token)
                #p.write('{}\t'.format(token).encode('utf-8')) 
                if params['include_lemma']:
                    lemma = text_tagged['lemma'][number]#[lem for lem in text_tagged['lemma'][number][num*3:num*3+3] if lem != '<PAD>']
                    token_plus_description.append('_')
                    token_plus_description.append(lemma)
                    #p.write('{}\t'.format('+'.join(lemma)).encode('utf-8')) 
                if params['include_pos']:
                    pos = text_tagged['pos'][number]#[pos for pos in text_tagged['pos'][number][num*3:num*3+3] if pos != '<PAD>']
                    token_plus_description.append('_')
                    token_plus_description.append(pos)#p.write('{}\t'.format('+'.join(pos)).encode('utf-8'))
                if params['include_morph']:
                    morph = text_tagged['morph'][number]#[morph for morph in text_tagged['morph'][number][num*3:num*3+3] if morph != '<PAD>']
                    token_plus_description.append('_')
                    token_plus_description.append(morph)#p.write('{}\t'.format('+'.join(morph)).encode('utf-8'))
                #space = r' '
                token_plus_description.append(' ')
                token_plus_description = ''.join(token_plus_description)
                
                
                p.write(token_plus_description.encode('utf-8'))
            
    return 'tagging process finished'











