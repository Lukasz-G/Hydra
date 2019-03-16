

import torch, os
from torch.autograd import Variable
from Communication import Info_exchange
import numpy as np
from Utils import generate_corpus_samples 
from Utils import loading_text_fragment, inverse_transform_pos_morph, find_multitoken_tags, find_multitag_tokens
from Utils import construct_input_repr, construct_output_repr1, construct_output_repr2, timeSince, read_list
from Utils import inverse_transform, count_lemmas_validation, find_unique_elements
#global use_cuda
#use_cuda = torch.cuda.is_available()
import gc
#global number_gpu
#number_gpu = torch.cuda.device_count()

from fuzzywuzzy import fuzz

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name_host = MPI.Get_processor_name()

'''Training and Validation class'''

class Training_and_Validation():
    def __init__(self,
        train_fragment=None,
        val_fragment=None,
        test_fragment=None,
        model=None,
        model_optimizer=None,
        criterion=None,
        char_vector_dict=None,
        include_lemma=False,
        include_morph=False,
        include_pos=False,
        max_len_tok=None,
        max_len_lemma=None,
        max_tag_len_seq=None,
        encoder_lemma=None,
        encoder_letters=None,
        encoder_pos=None,
        encoder_morph=None,
        dictionary=None,
        batch_size = None,
        print_freq = None,
        context_size = None,
        alpha=None,
        rank=None,
        size=None,
        active_workers = None,
        use_cuda=False,
        p_value=None,
        cellarium_folder = None,
        models_folder = None,
        corpus_stats_folder = None,
        n_iters = None
         
             ):
        
        self.train_fragment = train_fragment
        self.val_fragment = val_fragment
        self.test_fragment = test_fragment
        self.model = model
        self.model_optimizer = model_optimizer
        self.criterion = criterion
        self.char_vector_dict = char_vector_dict
        self.include_lemma = include_lemma
        self.include_pos = include_pos
        self.include_morph = include_morph
        self.encoder_letters=encoder_letters
        self.encoder_lemma=encoder_lemma
        if self.include_lemma:
            self.lemma_size=len(encoder_lemma.classes_)
        self.encoder_pos=encoder_pos
        self.encoder_morph=encoder_morph
        self.dictionary=dictionary
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.max_len_tok = max_len_tok
        self.max_len_lemma = max_len_lemma
        self.max_tag_len_seq = max_tag_len_seq
        self.context_size = context_size
        self.alpha = alpha
        self.rank = rank
        self.size = size
        self.active_workers = active_workers
        self.use_cuda = use_cuda
        self.p_value = p_value
        self.cellarium = cellarium_folder
        self.models_folder = models_folder
        self.corpus_stats_folder = corpus_stats_folder
        self.n_iters = n_iters
        
        
        ''' initialise exchange module '''
        
        self.exchange = Info_exchange(
            model=self.model,
            alpha=self.alpha,
            rank=self.rank,
            size=self.size,
            active_workers=self.active_workers,
            use_cuda=self.use_cuda
                      )
        
        
        
        
        '''load in the corpora for training, development and testing'''
       
        self.corpus_train = loading_text_fragment(self.train_fragment, self.include_lemma, self.include_pos, self.include_morph)
        self.corpus_dev = loading_text_fragment(self.val_fragment, self.include_lemma, self.include_pos, self.include_morph)
        self.corpus_test = loading_text_fragment(self.test_fragment, self.include_lemma, self.include_pos, self.include_morph)
        
        #print(self.corpus_train['lemma'][:10], self.corpus_dev['lemma'][:10])
        #quit()
        
        self.compute_loss = torch.nn.NLLLoss()#torch.nn.NLLLoss
        
        
        try:
            print('process no.{} will train its corpus fragment on local device no.{}/{}'.format(self.rank, torch.cuda.current_device(), torch.cuda.device_count()))
        except:
            pass
        
    #@profile
    def train(self, nth_iter = None, start_time=None):
        
        '''
        define initial variable for one epoch:
            -time
            -number of steps through corpus
            -print moment
            -exchange status for each worker
        
        '''
        
        #self.tr = tracker.SummaryTracker()
        #self.tr.print_diff() 
        self.start_time = start_time
        
        self.train_length = len(self.corpus_train['token'])
        if self.dictionary:
            self.train_length += len(self.dictionary['token'])
        self.steps_over_data_train = self.train_length//self.batch_size
        if self.train_length % self.batch_size != 0:
            self.steps_over_data_train += 1
        
        #print(self.steps_over_data_train, self.train_length, self.batch_size)
        #quit()
        self.print_moment = int(self.steps_over_data_train*self.print_freq)
        if self.print_moment == 0:
            self.print_moment = 1
        
        self.exchange.update_done = False
        
        loss_general_lemmatizer = 0.0
        loss_general_pos = 0.0
        loss_general_morph = 0.0
        
        #ignored_index_lemma = self.encoder_lemma.transform(['<PAD>']).tolist()[0]
        
        
        self.model.train()
          
        step = -1
        
        '''generate training samples from corpus'''
        
            
        
        for x in generate_corpus_samples(corpus=self.corpus_train, lemma=self.include_lemma, pos=self.include_pos, morph=self.include_morph, dictionary=self.dictionary,context_size=self.context_size, batch_size=self.batch_size):
            self.model.zero_grad()
            #self.tr.print_diff()   
            '''depending on specified options these variables either will contain samples or will be empty (with None)'''
            
            step += 1
            tokens, lemmas, poss, morphs, context_left, context_right = x
            
            
            
            '''vectorisation of tokens and their context words'''
            
            tokens_vectorised = construct_input_repr(tokens, self.encoder_letters, self.max_len_tok, context=False)
            
            if context_left != None and context_right != None:
                con_L_vectorised = construct_input_repr(context_left, self.encoder_letters, self.max_len_tok, context=True)
                con_R_vectorised = construct_input_repr(context_right, self.encoder_letters, self.max_len_tok, context=True)
            else:
                #context_left = []
                #for _ in range(len(tokens)):
                #    batch_frag = []
                #    for _ in range(self.context_size):
                #        random.seed(time.time())
                #        rnd_nb = random.randint(0,len(self.dictionary['token'])-1)
                #        batch_frag.append(self.dictionary['token'][rnd_nb])
                #    context_left.append(batch_frag)
                #context_right = []
                #for _ in range(len(tokens)):
                #    batch_frag = []
                #    for _ in range(self.context_size):
                #        random.seed(time.time())
                #        rnd_nb = random.randint(0,len(self.dictionary['token'])-1)
                #        batch_frag.append(self.dictionary['token'][rnd_nb])
                #    context_right.append(batch_frag)
                
                    
                #context_left = [[self.dictionary['token'][random.randint(0,len(self.dictionary['token'])] for _ in range(self.context_size)] for _ in range(len(tokens))]
                #context_right = [['<PAD>']*self.context_size for _ in range(len(tokens))] 
                con_L_vectorised = None#construct_input_repr(context_left, self.encoder_letters, self.max_len_tok, context=True)
                con_R_vectorised = None#construct_input_repr(context_right, self.encoder_letters, self.max_len_tok, context=True)
                
                
                
            '''vectorisation of output elements'''
            
            lemmas_vectorised, poss_vectorised, morphs_vectorised = None, None, None
            
            if self.include_lemma:
                lemmas_vectorised = construct_output_repr1(lemmas, self.encoder_lemma, max_len_lemma=self.max_len_lemma)
            if self.include_pos:
                poss_vectorised = construct_output_repr2(poss, self.encoder_pos, max_tag_len_seq=self.max_tag_len_seq)
            if self.include_morph:
                morphs_vectorised = construct_output_repr2(morphs, self.encoder_morph, max_tag_len_seq=self.max_tag_len_seq)
            
            loss_lemmatizer = 0.0
            loss_pos = 0.0
            loss_morph = 0.0
            
            
            
            
            
            '''run model'''
            
            output_lemmatizer, output_tagger_pos, output_tagger_morph, lemmas_embedded = \
            self.model(tokens=tokens_vectorised,
                context_left=con_L_vectorised, 
                context_right=con_R_vectorised,
                lemmas = lemmas_vectorised
                       )
            #print(self.model.return_output_embeddings)
            #quit()
            
            '''depending on specified options (include lemma-pos-morph) the model will be evaluated in regard to them '''
            
            values = np.array([self.include_lemma, self.include_pos, self.include_morph], dtype=bool)
            values = values.astype(int)
            
            '''variable 'summa' serves to retaining the graph for multiple backward runs if multiple options lemma-pos-morphs are specified'''
            
            summa = np.sum(values)
            
            if self.include_lemma:
                lemmas_vectorised_var = Variable(lemmas_vectorised.view(-1))
                if self.use_cuda:
                    lemmas_vectorised_var = lemmas_vectorised_var.cuda()
                
                #lemmas_embs = self.model.return_lemma_embeddings(lemmas_vectorised_var)
                #print(lemmas_vectorised_var.size(), output_lemmatizer.size())
                #quit()
                #model.return_output_embeddings
                loss_lemmatizer = self.compute_loss(output_lemmatizer, lemmas_vectorised_var)#lemmas_embs)
                #print(loss_lemmatizer)
                #quit()
                #loss_lemmatizer = torch.nn.functional.mse_loss(output_lemmatizer, lemmas_embedded)#,  ignore_index = self.encoder_lemma.transform(['<PAD>'])[0])
                #loss_lemmatizer = torch.nn.functional.nll_loss(output_lemmatizer, lemmas_vectorised_var)#,  ignore_index = self.encoder_lemma.transform(['<PAD>'])[0])
                
                summa -= 1
                if summa > 0:
                    retain_graph= True
                else:
                    retain_graph = False
                loss_lemmatizer.backward(retain_graph=retain_graph)
                loss_lemmatizer = loss_lemmatizer.item()#data[0]
                loss_general_lemmatizer += loss_lemmatizer
                
            if self.include_pos:
                poss_vectorised_var = Variable(poss_vectorised.view(-1))
                if self.use_cuda:
                    poss_vectorised_var = poss_vectorised_var.cuda()
                
                #poss_embs = self.model.return_pos_embeddings(poss_vectorised_var)
                
                loss_pos = self.compute_loss(output_tagger_pos, poss_vectorised_var)# poss_embs)#,ignore_index = self.encoder_pos.transform(['<PAD>'])[0])
                
                summa -= 1
                if summa > 0:
                    retain_graph= True
                else:
                    retain_graph = False
                loss_pos.backward(retain_graph=retain_graph)
                loss_pos = loss_pos.item()#data[0]
                loss_general_pos += loss_pos
                
            if self.include_morph:
                morphs_vectorised_var = Variable(morphs_vectorised.view(-1))
                if self.use_cuda:
                    morphs_vectorised_var = morphs_vectorised_var.cuda()
                
                #morph_embs = self.model.return_morph_embeddings(morphs_vectorised_var)
                loss_morph = self.compute_loss(output_tagger_morph, morphs_vectorised_var)#,ignore_index = self.encoder_morph.transform(['<PAD>'])[0])
                
                summa -= 1
                if summa > 0:
                    retain_graph= True
                else:
                    retain_graph = False
                loss_morph.backward(retain_graph=retain_graph)
                loss_morph = loss_morph.item()#data[0]
                loss_general_morph += loss_morph#
            
   
            
            '''clamping gradients'''
            
            #for p in self.coder.parameters():
            #    if p.grad is not None:
            #        p.grad.data.clamp_(-1.0,1.0)
            
            '''optimising model'''
                    
            self.model_optimizer.step()
            self.model.zero_grad()
            
            
                
            '''printing training information'''
             
            if step % self.print_moment == 0:
             
                percent = (self.train_length/self.batch_size*nth_iter + step + 1) / float(self.n_iters*self.train_length/self.batch_size)
                if percent == 0:
                    percent = 1
                time_sit = timeSince(self.start_time, percent)
                
                info_from_training = 'process no.{}: loss averaged at sequence no.{}/{} at epoch no.{} ({}%): lemmatizer - {}, pos-tagger - {}, morph-tagger - {},  time: {}'.format(self.rank, (step)+1, 
                                                                                    self.steps_over_data_train, nth_iter+1,
                                                                                percent,
                                                                                loss_lemmatizer/(step+1), 
                                                                                loss_pos/(step+1),
                                                                                loss_morph/(step+1),
                                                                                time_sit)
            
                print(info_from_training)
                
                '''writing training information in the info file'''
                with open(os.path.join(self.cellarium, "info_from_training.txt"), "a") as myfile:
                    myfile.write(info_from_training)
                    myfile.write('\n')
                    myfile.close()
          
            
            '''initialise exchange of the model's parameters'''
            a_draw = np.random.binomial(n=1, p=self.p_value, size=None)        
            if a_draw == 1:
                self.exchange.sender()
            else: 
                self.exchange.receiver()       
            self.exchange.update()
            
        '''save the number of trained epochs so far'''
        with open(os.path.join(self.models_folder, 'num_epochs_acc_proc_{}.txt'.format(self.rank)), "w") as myfile:
            myfile.write(str(nth_iter+1))
            myfile.close()
                
        '''save the model's parameters'''
        #file_for_model = os.path.join(self.cellarium, "model_saved_proc_{}".format(self.rank))
        file_for_model = open(os.path.join(self.models_folder, "model_saved_proc_{}".format(self.rank)), "w")
        torch.save(self.model.state_dict(), file_for_model)
        file_for_model.close()
                    
        '''save the model's optimiser'''
        file_for_model = open(os.path.join(self.models_folder, "optimiser_saved_proc_{}".format(self.rank)), "w")
        torch.save(self.model_optimizer.state_dict(), file_for_model)
        file_for_model.close()
        
        
        '''calculating average loss for one epoch'''
        
        if self.include_lemma:
            loss_lemma = loss_general_lemmatizer/self.steps_over_data_train
        else:
            loss_lemma = None 
        
        if self.include_pos:
            loss_pos = loss_general_pos/self.steps_over_data_train
        else:    
            loss_pos = None
        
        if self.include_morph:            
            loss_morph = loss_general_morph/self.steps_over_data_train
        else:    
            loss_morph = None
        
        gc.collect()
        
        #self.tr.print_diff()
        #del self.tr
        return loss_lemma, loss_pos, loss_morph
    
    
    '''a catch-up function to smoothly finish communication between workers'''
    def catch_up(self):
        
        
        if self.rank != 0:
            info = "fini no. {}".format(self.rank)
            req = comm.issend(info, dest=0, tag=1)
            while not MPI.Request.Test(req):
                self.exchange.receiver()
                self.exchange.update()
                
        else:
            for proc in xrange(len(self.active_workers) - 1):
                req = comm.irecv(source=MPI.ANY_SOURCE, tag=1)
                while not MPI.Request.Test(req):
                    self.exchange.receiver()
                    self.exchange.update()

        if self.rank == 0:
            msg = "further"
            for proc in self.active_workers:
                if proc == 0:
                    continue
                comm.ssend(msg, dest=proc, tag=1)
                print(msg)
        else:
            receiving_rests = True
            while receiving_rests:
                if comm.Iprobe(source=MPI.ANY_SOURCE, tag=0):
                    self.exchange.receiver()
                    self.exchange.update()
                elif comm.Iprobe(source=MPI.ANY_SOURCE, tag=1):
                    msg = comm.recv(buf=None, source=MPI.ANY_SOURCE, tag=1)
                    if msg == "further":
                        receiving_rests = False
                    del msg

    '''validation module'''          
    def valid_dev(self, nth_iter = 'Final', development=False):
        
        '''define basic variables as for the training module'''
        
        if development:
            corpus = self.corpus_dev
        else:
            corpus = self.corpus_test
        
        self.val_length = len(corpus['token'])
        if self.dictionary:
            self.val_length += len(self.dictionary['token'])
        
        self.steps_over_data_val = self.val_length//self.batch_size
        if self.val_length % self.batch_size != 0:
            self.steps_over_data_val += 1
        
        unique_tokens, unique_lemmas, multilabels_unique_lemmas = None, None, None
        if nth_iter != 'Final':
            unique_tokens = read_list(os.path.join(self.corpus_stats_folder, 'dev unique tokens.txt'))
            if self.include_lemma:
                unique_lemmas = read_list(os.path.join(self.corpus_stats_folder, 'dev_lemmas_unique.txt'))
                multilabels_unique_lemmas = read_list(os.path.join(self.corpus_stats_folder, 'multilabels_lemmas_unique_dev.txt'))
        else:
            unique_tokens = read_list(os.path.join(self.corpus_stats_folder, 'test unique tokens.txt'))
            if self.include_lemma:
                unique_lemmas = read_list(os.path.join(self.corpus_stats_folder, 'test_lemmas_unique.txt'))
                multilabels_unique_lemmas = read_list(os.path.join(self.corpus_stats_folder, 'multilabels_lemmas_unique_test.txt'))
            #pass
        
        results_lemmatizer = 0.0
        results_lemmatizer_multitoken = 0.0
        results_lemmatizer_multitag = 0.0
        results_lemmatizer_unique_tokens = 0.0
        results_lemmatizer_unique_lemmas = 0.0
        results_lemmatizer_unique_multilabel_lemmas = 0.0
        number_lemma_multitoken = 0.0
        number_lemma_multitag = 0.0
        
        
        results_pos_tagging = 0.0
        results_pos_tagging_multitoken = 0.0
        results_pos_tagging_multitag = 0.0
        number_pos_multitoken = 0.0
        number_pos_multitag = 0.0
        
        results_morph_tagging = 0.0
        results_morph_tagging_multitoken = 0.0
        results_morph_tagging_multitag = 0.0
        number_morph_multitoken = 0.0
        number_morph_multitag = 0.0
        
        loss_general_lemmatizer = 0.0
        loss_general_pos = 0.0
        loss_general_morph = 0.0
        
        #ignored_index_lemma = self.encoder_lemma.transform(['<PAD>']).tolist()[0]
        
        
        '''set the module in the evaluation mode'''
        self.model = self.model.eval()
        
        step = -1
        
        '''generating corpus samples and vectorising them as during the training'''
        
        for x in generate_corpus_samples(corpus=corpus, lemma=self.include_lemma, pos=self.include_pos, morph=self.include_morph, dictionary= self.dictionary,context_size=self.context_size, batch_size=self.batch_size):
        
            step += 1
            tokens, lemmas, poss, morphs, context_left, context_right = x
            
            
            lemmas_multitoken_idx = find_multitoken_tags(lemmas)
            poss_multitoken_idx = find_multitoken_tags(poss) # is necessary if counted below by inverse_transform_pos_morph?????
            morphs_multitoken_idx = find_multitoken_tags(morphs)
            
            lemmas_multitag_idx = find_multitag_tokens(lemmas)
            poss_multitag_idx = find_multitag_tokens(poss)
            morphs_multitag_idx = find_multitag_tokens(morphs)
        
            tokens_unique_idx = find_unique_elements(tokens, unique_tokens)
            if self.include_lemma:
                lemmas_unique_idx  = find_unique_elements(tokens, unique_lemmas)
                multilabels_lemmas_unique_idx = find_unique_elements(tokens, multilabels_unique_lemmas)
            #unique_tokens
            
            tokens_vectorised = construct_input_repr(tokens, self.encoder_letters, self.max_len_tok, context=False)
            
            if context_left != None and context_right != None:
                con_L_vectorised = construct_input_repr(context_left, self.encoder_letters, self.max_len_tok, context=True)
                con_R_vectorised = construct_input_repr(context_right, self.encoder_letters, self.max_len_tok, context=True)
            else:
                con_L_vectorised = None
                con_R_vectorised = None
            
            lemmas_vectorised, poss_vectorised, morphs_vectorised = None, None, None
            
            if self.include_lemma:
                lemmas_vectorised = construct_output_repr1(lemmas, self.encoder_lemma, max_len_lemma=self.max_len_lemma)
            if self.include_pos:
                poss_vectorised = construct_output_repr2(poss, self.encoder_pos, max_tag_len_seq=self.max_tag_len_seq)
            if self.include_morph:
                morphs_vectorised = construct_output_repr2(morphs, self.encoder_morph,max_tag_len_seq=self.max_tag_len_seq)
            
            
            output_lemmatizer, output_tagger_pos, output_tagger_morph, lemmas_embedded = \
            self.model(tokens=tokens_vectorised,
                context_left=con_L_vectorised, 
                context_right=con_R_vectorised,
                lemmas = lemmas_vectorised
                       )
            
            '''calculating the loss for different options lemma-pos-morph'''
            
            if self.include_lemma:
                lemmas_vectorised_var = Variable(lemmas_vectorised.view(-1))
                if self.use_cuda:
                    lemmas_vectorised_var = lemmas_vectorised_var.cuda()
                #lemmas_embs = self.model.return_lemma_embeddings(lemmas_vectorised_var)
                
                loss_lemmatizer = self.compute_loss(output_lemmatizer, lemmas_vectorised_var)#,  ignore_index = self.encoder_lemma.transform(['<PAD>'])[0])
                
                loss_lemmatizer = loss_lemmatizer.data.item()#tolist()
                
                loss_general_lemmatizer += loss_lemmatizer
                
            if self.include_pos:
                poss_vectorised_var = Variable(poss_vectorised.view(-1))
                if self.use_cuda:
                    poss_vectorised_var = poss_vectorised_var.cuda()
                #pos_embs = self.model.return_pos_embeddings(poss_vectorised_var)
                loss_pos = self.compute_loss(output_tagger_pos, poss_vectorised_var)#,ignore_index = self.encoder_pos.transform(['<PAD>'])[0])
                
                loss_pos = loss_pos.data.tolist()
                
                loss_general_pos += loss_pos
                
            if self.include_morph:
                morphs_vectorised_var = Variable(morphs_vectorised.view(-1))
                if self.use_cuda:
                    morphs_vectorised_var = morphs_vectorised_var.cuda()
                #morph_embs = self.model.return_morph_embeddings(morphs_vectorised_var)
                loss_morph = self.compute_loss(output_tagger_morph, morphs_vectorised_var)#, ignore_index = self.encoder_morph.transform(['<PAD>'])[0])
                
                loss_morph = loss_morph.data.tolist()
                
                loss_general_morph += loss_morph

            '''
            calculating correctness of predictions 
            and
            saving exemplary samples from evaluation for different options lemma-pos-morph
            '''
            
            if self.include_lemma:   
               
            
                #results = self.model.most_similar_lemma(output_lemmatizer).data.cpu().view(-1,self.max_len_lemma).numpy().tolist()
                results = output_lemmatizer.view(-1,self.max_len_lemma,self.lemma_size).data.cpu().topk(1,dim=2)[1].contiguous().view(-1,self.max_len_lemma).numpy().tolist()
                
                
                results = [result.replace('<PAD>','') for result in inverse_transform(results, self.encoder_lemma)]
              
                general_count, multitoken_count, multitag_count, \
                tokens_unique_count, lemmas_unique_count, multilabel_lemmas_unique_count\
                = count_lemmas_validation(results, lemmas, 
                                          lemmas_multitoken_idx, lemmas_multitag_idx,
                                          tokens_unique_idx, lemmas_unique_idx, 
                                          multilabels_lemmas_unique_idx
                                          )
                
                results_lemmatizer += general_count
                results_lemmatizer_multitoken += multitoken_count
                results_lemmatizer_multitag += multitag_count
                results_lemmatizer_unique_tokens += tokens_unique_count
                results_lemmatizer_unique_lemmas += lemmas_unique_count
                results_lemmatizer_unique_multilabel_lemmas += multilabel_lemmas_unique_count
                
                number_lemma_multitoken += len(lemmas_multitoken_idx)
                number_lemma_multitag += len(lemmas_multitag_idx)
                
                if step == 0:
                    
                    myfile = open(os.path.join(self.cellarium, "results_from_eval.txt"), "a")
                    
                    for num, lemma in enumerate(lemmas):
                        
                        if True:# lemma != results[num]:
                        
                            phrases ='{};{}  ---->>>  {}'.format(tokens[num].encode('utf-8'),lemma.encode('utf-8'), results[num].encode('utf-8'))
                        
                            myfile.write('lemmatiser results at epoch no. {}:'.format(nth_iter))
                            myfile.write(phrases)
                            myfile.write('\n')
                    myfile.close()
                        
            if self.include_pos:
               
                pos_original = poss_vectorised.view(-1,self.max_tag_len_seq).numpy().tolist()
                #pos_predicted = self.model.most_similar_pos(output_tagger_pos).data.cpu().view(-1,self.max_tag_len_seq).numpy().tolist()
                pos_predicted = output_tagger_pos.data.cpu().topk(1,dim=1)[1].view(-1,self.max_tag_len_seq).numpy().tolist()
            
                
                pos_counts, pos_predicted,pos_original, multi_token_result, \
                multi_token_original, multi_tag_result, multi_tag_original \
                = inverse_transform_pos_morph(pos_predicted, pos_original, self.encoder_pos)
            
                results_pos_tagging_multitoken += multi_token_result
                results_pos_tagging_multitag += multi_tag_result
                results_pos_tagging += pos_counts
                number_pos_multitoken += multi_token_original
                number_pos_multitag += multi_tag_original
                
                if step == 0:
                    myfile = open(os.path.join(self.cellarium, "results_from_eval.txt"), "a")
                    
                    for num, pos in enumerate(pos_original):
                        
                        
                        
                        phrases ='{}  ---->>>  {}'.format(pos.encode('utf-8'), pos_predicted[num].encode('utf-8'))
                        
                        myfile.write('pos tagging results at epoch no. {}:'.format(nth_iter))
                        myfile.write(phrases)
                        myfile.write('\n')
                    myfile.close()
                    
            if self.include_morph:   
               
                morph_original = morphs_vectorised.view(-1,self.max_tag_len_seq).numpy().tolist()
                #morph_predicted = self.model.most_similar_morph(output_tagger_morph).data.cpu().view(-1,self.max_tag_len_seq).numpy().tolist()
                morph_predicted = output_tagger_morph.data.cpu().topk(1,dim=1)[1].view(-1,self.max_tag_len_seq).numpy().tolist()#.numpy()
                
                morph_counts, morph_predicted, morph_original, multi_token_result, \
                multi_token_original, multi_tag_result, multi_tag_original \
                = inverse_transform_pos_morph(morph_predicted, morph_original, self.encoder_morph)
                
                results_morph_tagging_multitoken += multi_token_result
                results_morph_tagging_multitag += multi_tag_result
                results_morph_tagging += morph_counts
                number_morph_multitoken += multi_token_original
                number_morph_multitag += multi_tag_original
                
          
                if step == 0:
                    myfile = open(os.path.join(self.cellarium, "results_from_eval.txt"), "a")
                    
                    for num, morph in enumerate(morph_original):
                        
                        phrases ='{}  ---->>>  {}'.format(morph.encode('utf-8'), morph_predicted[num].encode('utf-8'))
                        
                        myfile.write('morph tagging results at epoch no. {}:'.format(nth_iter))
                        myfile.write(phrases)
                        myfile.write('\n')
                    
                    myfile.close()
        
        
        
        
        
        results_lemma, results_pos, results_morph, loss_lemma, loss_pos, loss_morph = None, None, None, None, None, None
        
        
        '''divide results and loss through final number of steps'''
        
        
        if self.include_lemma:
            results_lemma = results_lemmatizer/self.steps_over_data_val
            loss_lemma = loss_general_lemmatizer/self.steps_over_data_val
            
            if number_lemma_multitag:
                results_lemma_multitag = results_lemmatizer_multitag/number_lemma_multitag
            else:
                results_lemma_multitag= 0.0
            
            if number_lemma_multitoken:
                results_lemma_multitoken = results_lemmatizer_multitoken/number_lemma_multitoken
            else:
                results_lemma_multitoken = 0.0
            
            if unique_tokens:
                results_lemma_unique_tokens = results_lemmatizer_unique_tokens/len(unique_tokens)
            else:
                results_lemma_unique_tokens = 0.0
            
            if unique_lemmas:
                results_lemma_unique_lemmas = results_lemmatizer_unique_lemmas/len(unique_lemmas)
            else: 
                results_lemma_unique_lemmas = 0.0
            
            if multilabels_unique_lemmas:
                results_lemma_unique_multilabel_lemmas = results_lemmatizer_unique_multilabel_lemmas/len(multilabels_unique_lemmas)
            else:
                results_lemma_unique_multilabel_lemmas = 0.0
            
            l = [results_lemma, results_lemma_multitag, results_lemma_multitoken,\
                results_lemma_unique_tokens, results_lemma_unique_lemmas, \
                results_lemma_unique_multilabel_lemmas]
        else:
            l = None
        if self.include_pos:
            results_pos = results_pos_tagging/self.steps_over_data_val
            loss_pos = loss_general_pos/self.steps_over_data_val
            if number_pos_multitag:
                results_pos_multitag = results_pos_tagging_multitag/number_pos_multitag
            else:
                results_pos_multitag = 0.0 
            if number_pos_multitoken:
                results_pos_multitoken = results_pos_tagging_multitoken/number_pos_multitoken
            else:
                results_pos_multitoken = 0.0
            
            p = [results_pos, results_pos_multitag, results_pos_multitoken]  
        else:
            p = None      
        if self.include_morph:
            results_morph = results_morph_tagging/self.steps_over_data_val
            loss_morph = loss_general_morph/self.steps_over_data_val
            if number_morph_multitag:
                results_morph_multitag = results_morph_tagging_multitag/number_morph_multitag
            else:
                results_morph_multitag = 0.0 
            if number_morph_multitoken:
                results_morph_multitoken = results_morph_tagging_multitoken/number_morph_multitoken
            else:
                results_morph_multitoken = 0.0
            
            m = [results_morph, results_morph_multitag, results_morph_multitoken]   
            
        else:
            m = None
        
        gc.collect()
        
        return l, p, m, loss_lemma, loss_pos, loss_morph
    