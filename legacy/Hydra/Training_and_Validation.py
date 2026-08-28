

import torch, os, random
from torch.autograd import Variable
from .Communication import Info_exchange
import numpy as np
from .Utils import generate_corpus_samples 
from .Utils import loading_text_fragment, inverse_transform_pos_morph, find_multitoken_tags, find_multitag_tokens
from .Utils import construct_input_repr, construct_output_repr1, construct_output_repr2, timeSince, read_list
from .Utils import inverse_transform, count_lemmas_validation, find_unique_elements
#global use_cuda
#use_cuda = torch.cuda.is_available()
import gc
#global number_gpu
#number_gpu = torch.cuda.cuda_count()

from fuzzywuzzy import fuzz

import logging, sys
logging.basicConfig(filename='example.log',level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

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

        self.teacher_forcing_ratio = 1.0

        self.SOS_lemma, self.EOS_lemma, self.PAD_lemma = self.encoder_lemma.transform(["<SOS>","<EOS>","<PAD>"])
        self.SOS_pos, self.EOS_pos, self.PAD_pos = self.encoder_pos.transform(["<SOS>","<EOS>","<PAD>"])
        self.SOS_morph, self.EOS_morph, self.PAD_morph = self.encoder_morph.transform(["<SOS>","<EOS>","<PAD>"])
        
        # transform output label weights into PyTorch tensors and optionally put them on a cuda device
        if self.encoder_lemma is not None:
            self.lemma_label_weight = torch.from_numpy(self.encoder_lemma.weights_).float()
            if self.use_cuda:
                self.lemma_label_weight = self.lemma_label_weight.to('cuda:{}'.format(rank))
        if self.encoder_pos is not None:
            self.pos_label_weight = torch.from_numpy(self.encoder_pos.weights_).float()
            if self.use_cuda:
                self.pos_label_weight = self.pos_label_weight.to('cuda:{}'.format(rank))
        if self.encoder_morph is not None:
            self.morph_label_weight = torch.from_numpy(self.encoder_morph.weights_).float()
            if self.use_cuda:
                self.morph_label_weight = self.morph_label_weight.to('cuda:{}'.format(rank))


        
        
        #quit()
        
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
        
        from tqdm import tqdm   

        pbar = tqdm(total=self.steps_over_data_train) 
        
        for x in generate_corpus_samples(corpus=self.corpus_train, lemma=self.include_lemma, pos=self.include_pos, morph=self.include_morph, dictionary=self.dictionary,context_size=self.context_size, batch_size=self.batch_size):
            self.model.zero_grad()
            #self.tr.print_diff()   
            '''depending on specified options these variables either will contain samples or will be empty (with None)'''
            
            step += 1
            tokens, lemmas, poss, morphs, context_left, context_right = x
            
            
            
            '''vectorisation of tokens and their context words'''
            
            tokens_vectorised = construct_input_repr(tokens, self.encoder_letters, self.max_len_tok, context=False)
            #print(tokens_vectorised.size())
            #quit()
            
            if context_left != None and context_right != None:
                con_L_vectorised = construct_input_repr(context_left, self.encoder_letters, self.max_len_tok, context=True)
                # reverse !!!
                
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
            
            if self.use_cuda:
                lemmas_vectorised = lemmas_vectorised.to('cuda:{}'.format(rank))
            if self.use_cuda:
                poss_vectorised = poss_vectorised.to('cuda:{}'.format(rank))
            if self.use_cuda:
                morphs_vectorised = morphs_vectorised.to('cuda:{}'.format(rank))
            
            
            #print(lemmas_vectorised)
            #print(poss_vectorised)
            
            #quit()
            loss_lemmatizer = 0.0
            loss_pos = 0.0
            loss_morph = 0.0
            
            
            if self.use_cuda:
                tokens_vectorised = tokens_vectorised.to('cuda:{}'.format(rank))
                con_L_vectorised = con_L_vectorised.to('cuda:{}'.format(rank))
                con_R_vectorised = con_R_vectorised.to('cuda:{}'.format(rank))

            
            
            '''run model'''
            
            encoder_output = \
            self.model.encode(tokens=tokens_vectorised,
                context_left=con_L_vectorised, 
                context_right=con_R_vectorised,
                       )
            #print(output_encoded.size())
            #quit()
            
            

            '''depending on specified options (include lemma-pos-morph) the model will be evaluated in regard to them '''
            
            values = np.array([self.include_lemma, self.include_pos, self.include_morph], dtype=bool)
            values = values.astype(int)
            
            '''variable 'summa' serves to retaining the graph for multiple backward runs if multiple options lemma-pos-morphs are specified'''
            
            summa = np.sum(values)
            
            if self.include_lemma:
                #lemmas_vectorised_var = Variable(lemmas_vectorised.view(-1))
                
                #print(lemmas_vectorised.size())
                #quit()
                

                loss_lemmatizer = 0
                decoder_hidden = self.model.init_hidden_lemmas(lemmas_vectorised.size()[0])
                decoder_input = torch.tensor([self.SOS_lemma]*lemmas_vectorised.size()[0])
                if self.use_cuda:
                    if self.model.rnn_attn_lemmas.rnn_type == 'lstm':
                        decoder_hidden = (decoder_hidden[0].to('cuda:{}'.format(rank)), decoder_hidden[1].to('cuda:{}'.format(rank)))
                    else:
                        decoder_hidden = decoder_hidden.to('cuda:{}'.format(rank))
                    decoder_input = decoder_input.to('cuda:{}'.format(rank))


                
                encoder_output_lemma = torch.clone(encoder_output)
                #reshape to 1 x B x D_out 
                encoder_output_lemma = encoder_output_lemma.unsqueeze(0) 
                decoder_hidden = encoder_output_lemma#, decoder_hidden[1]
                
                for indx_position in range(lemmas_vectorised.size()[1]):
                    #print(indx_position)

                    target_tensor = lemmas_vectorised[:,indx_position]
                    if self.PAD_lemma in target_tensor:
                        indcs_out = (target_tensor != self.PAD_lemma).nonzero().reshape(-1) 
                        #print(indcs_out, indcs_out.nelement())
                        if indcs_out.nelement() == 0:
                            break
                        #print(indcs_out)   
                        target_tensor = target_tensor[indcs_out]
                        if self.model.rnn_attn_lemmas.rnn_type == 'lstm':
                            decoder_hidden = (decoder_hidden[0][:,indcs_out], decoder_hidden[1][:,indcs_out])
                        else:
                            decoder_hidden = decoder_hidden[:,indcs_out]
                        encoder_output_lemma = encoder_output_lemma[:,indcs_out]
                        decoder_input = decoder_input[indcs_out]
                        lemmas_vectorised = lemmas_vectorised[indcs_out]
                    if target_tensor.size(0) == 0:
                        break
                    #print(decoder_input.size(), target_tensor.size())

                    
                    if self.use_cuda:
                        target_tensor = target_tensor.to('cuda:{}'.format(rank))
                    
                    #if indcs_out is not None:
                    #    lemmas_vectorised = lemmas_vectorised[indcs_out]
                    #    indcs_out  = None
                
                    use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

                    if use_teacher_forcing:
                        # Teacher forcing: Feed the target as the next input
                        #for di in range(target_length):
                        #print(decoder_input.size(), decoder_hidden.size(), encoder_output_lemma.size())
                        decoder_output, decoder_hidden = self.model.rnn_attn_lemmas(
                                decoder_input, decoder_hidden, encoder_output_lemma)
                        #print(decoder_output.size(), target_tensor.size(), "use")
                        loss_lemmatizer += torch.nn.functional.nll_loss(decoder_output, target_tensor, weight=self.lemma_label_weight)
                        decoder_input = target_tensor.detach()  # Teacher forcing


                    else:
                        # Without teacher forcing: use its own predictions as the next input
                        #print(decoder_input.size(), decoder_hidden.size(), encoder_output.size(), "before")
                        decoder_output, decoder_hidden = self.model.rnn_attn_lemmas(
                                decoder_input, decoder_hidden, encoder_output_lemma)
                        topv, topi = decoder_output.topk(1)
                        decoder_input = topi.detach()  # detach from history as input
                        #print(decoder_input.size(), decoder_hidden.size(), encoder_output.size(), target_tensor.size(), "after")
                        loss_lemmatizer += torch.nn.functional.nll_loss(decoder_output.squeeze(), target_tensor, weight=self.lemma_label_weight)
                        if self.EOS_lemma in decoder_input:
                            indcs_out = (decoder_input != self.EOS_lemma).nonzero().reshape(-1)
                            #print(indcs_out, decoder_input, "indcs out")
                            #quit()
                            #print(decoder_input.size(), decoder_hidden.size(), encoder_output.size())
                            decoder_input = decoder_input[indcs_out]
                            if self.model.rnn_attn_lemmas.rnn_type == 'lstm':
                                decoder_hidden = (decoder_hidden[0][:,indcs_out], decoder_hidden[1][:,indcs_out])
                            else:
                                decoder_hidden = decoder_hidden[:,indcs_out]
                            encoder_output_lemma = encoder_output_lemma[:,indcs_out]
                            lemmas_vectorised = lemmas_vectorised[indcs_out]
                            
                            #reak

                
                #quit()
                            
                
                
                #lemmas_embs = self.model.return_lemma_embeddings(lemmas_vectorised_var)
                #print(lemmas_vectorised_var.size(), output_lemmatizer.size())
                #quit()
                #model.return_output_embeddings
                #loss_lemmatizer = self.compute_loss(output_lemmatizer, lemmas_vectorised_var)#lemmas_embs)
                #print(loss_lemmatizer)
                #quit()
                #loss_lemmatizer = torch.nn.functional.mse_loss(output_lemmatizer, lemmas_embedded)#,  ignore_index = self.encoder_lemma.transform(['<PAD>'])[0])
                #loss_lemmatizer = torch.nn.functional.nll_loss(output_lemmatizer, lemmas_vectorised_var)#,  ignore_index = self.encoder_lemma.transform(['<PAD>'])[0])
                #print('finished lemma loss summing')
                summa -= 1
                if summa > 0:
                    retain_graph= True
                else:
                    retain_graph = False
                loss_lemmatizer.backward(retain_graph=retain_graph)
                loss_lemmatizer = loss_lemmatizer.item()#data[0]
                loss_general_lemmatizer += loss_lemmatizer
            #print('passed lemmas')    
            if self.include_pos:

                loss_pos = 0
                decoder_hidden = self.model.init_hidden_pos(poss_vectorised.size()[0])
                decoder_input = torch.tensor([self.SOS_pos]*poss_vectorised.size()[0])
                if self.use_cuda:
                    if self.model.rnn_attn_pos.rnn_type == 'lstm':
                        decoder_hidden = (decoder_hidden[0].to('cuda:{}'.format(rank)),decoder_hidden[1].to('cuda:{}'.format(rank)))
                    else:
                        decoder_hidden = decoder_hidden.to('cuda:{}'.format(rank))
                    decoder_input = decoder_input.to('cuda:{}'.format(rank))

                encoder_output_pos = torch.clone(encoder_output)
                encoder_output_pos = encoder_output_pos.unsqueeze(0) 
                decoder_hidden = encoder_output_pos#, decoder_hidden[1]


                #print(poss_vectorised, poss_vectorised.size())
                for indx_position in range(poss_vectorised.size()[1]):

                    target_tensor = poss_vectorised[:,indx_position]

                    if self.PAD_pos in target_tensor:
                        indcs_out = (target_tensor != self.PAD_pos).nonzero().reshape(-1)
                        if indcs_out.nelement() == 0:
                            break
                        target_tensor = target_tensor[indcs_out]
                        if self.model.rnn_attn_pos.rnn_type == 'lstm':
                            decoder_hidden = (decoder_hidden[0][:,indcs_out], decoder_hidden[1][:,indcs_out])
                        else:
                            decoder_hidden = decoder_hidden[:,indcs_out]
                        decoder_input = decoder_input[indcs_out]
                        encoder_output_pos = encoder_output_pos[:,indcs_out]
                        poss_vectorised = poss_vectorised[indcs_out]
                    if target_tensor.size(0) == 0:
                        break

                    
                    
                    if self.use_cuda:
                        target_tensor = target_tensor.to('cuda:{}'.format(rank))

                    
                    
                    use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

                    if use_teacher_forcing:
                        # Teacher forcing: Feed the target as the next input
                        #for di in range(target_length):
                        #print(decoder_input.size(), decoder_hidden.size(), encoder_output_pos.size())
                        decoder_output, decoder_hidden = self.model.rnn_attn_pos(
                                decoder_input, decoder_hidden, encoder_output_pos)
                        #print(decoder_output.size(), target_tensor.size())
                        loss_pos += torch.nn.functional.nll_loss(decoder_output, target_tensor, weight=self.pos_label_weight)
                        decoder_input = target_tensor.detach()
                        
                        #decoder_input = target_tensor  # Teacher forcing

                    else:
                        # Without teacher forcing: use its own predictions as the next input
                        #print(decoder_input.size(), decoder_hidden.size(), encoder_output.size(), "before")
                        decoder_output, decoder_hidden = self.model.rnn_attn_pos(
                                decoder_input, decoder_hidden, encoder_output_pos)
                        topv, topi = decoder_output.topk(1)
                        decoder_input = topi.detach()  # detach from history as input
                        #print(decoder_output.size(), target_tensor.size(), 'posss')
                        #print(decoder_input.size(), decoder_hidden.size(), encoder_output.size(), target_tensor.size(), "after")
                        loss_pos += torch.nn.functional.nll_loss(decoder_output, target_tensor, weight=self.pos_label_weight)
                        if self.EOS_pos in decoder_input:
                            indcs_out = (decoder_input != self.EOS_pos).nonzero().reshape(-1)
                            #print((decoder_input == self.EOS).nonzero()[0])
                            #print(decoder_input.size(), decoder_hidden.size(), encoder_output.size())
                            decoder_input = decoder_input[indcs_out]
                            if self.model.rnn_attn_pos.rnn_type == 'lstm':
                                decoder_hidden = (decoder_hidden[0][:,indcs_out], decoder_hidden[1][:,indcs_out])
                            else:
                                decoder_hidden = decoder_hidden[:,indcs_out]
                            encoder_output_pos = encoder_output_pos[:,indcs_out]
                            poss_vectorised = poss_vectorised[indcs_out]
                            
                            #break
                summa -= 1
                if summa > 0:
                    retain_graph= True
                else:
                    retain_graph = False
                loss_pos.backward(retain_graph=retain_graph)
                loss_pos = loss_pos.item()#data[0]
                loss_general_pos += loss_pos
            #print('passed pos')    
            
            if self.include_morph:
                loss_morph = 0
                decoder_hidden = self.model.init_hidden_morph(morphs_vectorised.size()[0])
                decoder_input = torch.tensor([self.SOS_morph]*morphs_vectorised.size()[0])
                if self.use_cuda:
                    if self.model.rnn_attn_morph.rnn_type == 'lstm':
                        decoder_hidden = (decoder_hidden[0].to('cuda:{}'.format(rank)),decoder_hidden[1].to('cuda:{}'.format(rank)))
                    else:
                        decoder_hidden = decoder_hidden.to('cuda:{}'.format(rank))
                    decoder_input = decoder_input.to('cuda:{}'.format(rank))

                encoder_output_morph = torch.clone(encoder_output)
                encoder_output_morph = encoder_output_morph.unsqueeze(0) 
                decoder_hidden = encoder_output_morph#, decoder_hidden[1]
                
                for indx_position in range(morphs_vectorised.size()[1]):

                    target_tensor = morphs_vectorised[:,indx_position]
                    
                    if self.PAD_morph in target_tensor:
                        indcs_out = (target_tensor != self.PAD_morph).nonzero().reshape(-1)
                        if indcs_out.nelement() == 0:
                            break  
                        target_tensor = target_tensor[indcs_out]
                        if self.model.rnn_attn_morph.rnn_type == 'lstm':
                            decoder_hidden = (decoder_hidden[0][:,indcs_out], decoder_hidden[1][:,indcs_out])
                        else:
                            decoder_hidden = decoder_hidden[:,indcs_out]
                        decoder_input = decoder_input[indcs_out]
                        encoder_output_morph = encoder_output_morph[:,indcs_out]
                        morphs_vectorised = morphs_vectorised[indcs_out]
                    if target_tensor.size(0) == 0:
                        break
                    
                    if self.use_cuda:
                        target_tensor = target_tensor.to('cuda:{}'.format(rank))

                
                    #if indcs_out is not None:
                    #    morphs_vectorised = morphs_vectorised[indcs_out]
                    #    indcs_out = None
                    
                    use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

                    if use_teacher_forcing:
                        # Teacher forcing: Feed the target as the next input
                        #for di in range(target_length):
                        decoder_output, decoder_hidden = self.model.rnn_attn_morph(
                                decoder_input, decoder_hidden, encoder_output_morph)
                        #print(decoder_output.size(), target_tensor.size())
                        loss_morph += torch.nn.functional.nll_loss(decoder_output, target_tensor, weight=self.morph_label_weight)
                        decoder_input = target_tensor.detach()  # Teacher forcing

                    else:
                        # Without teacher forcing: use its own predictions as the next input
                        #print(decoder_input.size(), decoder_hidden.size(), encoder_output.size(), "before")
                        decoder_output, decoder_hidden = self.model.rnn_attn_morph(
                                decoder_input, decoder_hidden, encoder_output_morph)
                        topv, topi = decoder_output.topk(1)
                        decoder_input = topi.squeeze().detach()  # detach from history as input
                        #print(decoder_output.size(), target_tensor.size())
                        #print(decoder_input.size(), decoder_hidden.size(), encoder_output.size(), target_tensor.size(), "after")
                        loss_morph += torch.nn.functional.nll_loss(decoder_output, target_tensor, weight=self.morph_label_weight)
                        if self.EOS_morph in decoder_input:
                            indcs_out = (decoder_input != self.EOS_morph).nonzero().reshape(-1)
                            #print((decoder_input == self.EOS).nonzero()[0])
                            #print(decoder_input.size(), decoder_hidden.size(), encoder_output.size())
                            decoder_input = decoder_input[indcs_out]
                            if self.model.rnn_attn_morph.rnn_type == 'lstm':
                                decoder_hidden = (decoder_hidden[0][:,indcs_out], decoder_hidden[1][:,indcs_out])
                            else:
                                decoder_hidden = decoder_hidden[:,indcs_out]
                            encoder_output_morph = encoder_output_morph[:,indcs_out]
                            morphs_vectorised = morphs_vectorised[indcs_out]
                            
                            #break
                
                summa -= 1
                if summa > 0:
                    retain_graph= True
                else:
                    retain_graph = False
                loss_morph.backward(retain_graph=retain_graph)
                loss_morph = loss_morph.item()#data[0]
                loss_general_morph += loss_morph#
            
            #print('passed morph')
            
            '''clamping gradients'''
            
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-1.0,1.0)
            
            '''optimising model'''
                    
            self.model_optimizer.step()
            self.model.zero_grad()
            
            
                
            '''printing training information'''
            pbar.update(1)
            if False:# step % self.print_moment == 0:
             
                percent = (self.train_length/self.batch_size*nth_iter + step + 1) / float(self.n_iters*self.train_length/self.batch_size)
                if percent == 0:
                    percent = 1
                time_sit = timeSince(self.start_time, percent)
                
                info_from_training = 'process no.{}: loss averaged at step no.{}/{} at epoch no.{} ({}%): lemmatizer - {}, pos-tagger - {}, morph-tagger - {},  time: {}'.format(self.rank, (step)+1, 
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
                #logging.info('sending something')
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
        #file_for_model = open(os.path.join(self.models_folder, "model_saved_proc_{}".format(self.rank)), "wb")
        torch.save(self.model.state_dict(), os.path.join(self.models_folder, "model_saved_proc_{}".format(self.rank)))
        #file_for_model.close()
                    
        '''save the model's optimiser'''
        #file_for_model = open(os.path.join(self.models_folder, "optimiser_saved_proc_{}".format(self.rank)), "w")
        torch.save(self.model_optimizer.state_dict(), os.path.join(self.models_folder, "optimiser_saved_proc_{}".format(self.rank)))
        #file_for_model.close()
        
        
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
        
        print(loss_lemma, loss_pos, loss_morph)
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
            for proc in range(len(self.active_workers) - 1):
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

            if self.use_cuda:
                tokens_vectorised = tokens_vectorised.to('cuda:{}'.format(rank))
                con_L_vectorised = con_L_vectorised.to('cuda:{}'.format(rank))
                con_R_vectorised = con_R_vectorised.to('cuda:{}'.format(rank))
            
            
            encoder_output = \
            self.model.encode(tokens=tokens_vectorised,
                context_left=con_L_vectorised, 
                context_right=con_R_vectorised
                       )
            
            '''calculating the loss for different options lemma-pos-morph'''
            
            if self.include_lemma:
                loss_lemmatizer = 0
                decoder_hidden = self.model.init_hidden_lemmas(lemmas_vectorised.size()[0])
                decoder_input = torch.tensor([self.SOS_lemma]*lemmas_vectorised.size()[0])
                if self.use_cuda:
                    if self.model.rnn_attn_lemmas.rnn_type == 'lstm':
                        decoder_hidden = (decoder_hidden[0].to('cuda:{}'.format(rank)), decoder_hidden[1].to('cuda:{}'.format(rank)))
                    else:
                        decoder_hidden = decoder_hidden.to('cuda:{}'.format(rank))
                    decoder_input = decoder_input.to('cuda:{}'.format(rank))

                 
                encoder_output_lemmas = torch.clone(encoder_output)
                encoder_output_lemmas = encoder_output_lemmas.unsqueeze(0)
                decoder_hidden = encoder_output_lemmas#, decoder_hidden[1]
                
                output_lemmatizer = torch.Tensor([self.PAD_lemma]).expand(lemmas_vectorised.size()).long().clone()
                output_lemmatizer_indcs = torch.LongTensor(list(range(output_lemmatizer.size(0))))
                
                for indx_position in range(lemmas_vectorised.size()[1]):

                    target_tensor = lemmas_vectorised[:,indx_position]
                    if self.use_cuda:
                        target_tensor = target_tensor.to('cuda:{}'.format(rank))

                    if self.PAD_lemma in target_tensor:
                        indcs_out = (target_tensor != self.PAD_lemma).nonzero().reshape(-1)
                        if indcs_out.nelement() == 0:
                            break  
                        target_tensor = target_tensor[indcs_out]
                        if self.model.rnn_attn_lemmas.rnn_type == 'lstm':
                            decoder_hidden = (decoder_hidden[0][:,indcs_out], decoder_hidden[1][:,indcs_out])
                        else:
                            decoder_hidden = decoder_hidden[:,indcs_out]
                        encoder_output_lemmas = encoder_output_lemmas[:,indcs_out]
                        decoder_input = decoder_input[indcs_out]
                        lemmas_vectorised = lemmas_vectorised[indcs_out.to('cpu')]
                        output_lemmatizer_indcs = output_lemmatizer_indcs[indcs_out.to('cpu')]
                    if target_tensor.size(0) == 0:
                        break

                    decoder_output, decoder_hidden = self.model.rnn_attn_lemmas(
                                decoder_input, decoder_hidden, encoder_output_lemmas)
                        #print(decoder_output.size(), target_tensor.size())
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach()  # detach from history as input
                    #if indcs_out is not None:
                    output_lemmatizer[output_lemmatizer_indcs,indx_position] = decoder_input.data.cpu().reshape(-1)
                    #else:
                    #    output_lemmatizer[:,indx_position] = decoder_input.data.cpu().reshape(-1)
                    #print(decoder_output.size(), target_tensor.size())
                    loss_lemmatizer += torch.nn.functional.nll_loss(decoder_output, target_tensor, weight=self.lemma_label_weight)

                
                loss_general_lemmatizer += loss_lemmatizer.item()
                
            if self.include_pos:

                loss_pos = 0
                decoder_hidden = self.model.init_hidden_pos(poss_vectorised.size()[0])
                decoder_input = torch.tensor([self.SOS_pos]*poss_vectorised.size()[0])
                if self.use_cuda:
                    if self.model.rnn_attn_pos.rnn_type == 'lstm':
                        decoder_hidden = (decoder_hidden[0].to('cuda:{}'.format(rank)), decoder_hidden[1].to('cuda:{}'.format(rank)))
                    else:
                        decoder_hidden = decoder_hidden.to('cuda:{}'.format(rank))
                    decoder_input = decoder_input.to('cuda:{}'.format(rank))
                #decoder_hidden = encoder_output.unsqueeze(0)#, decoder_hidden[1]
                encoder_output_pos = torch.clone(encoder_output)
                encoder_output_pos = encoder_output_pos.unsqueeze(0)
                decoder_hidden = encoder_output_pos#, decoder_hidden[1]
                poss_vectorised_orig = poss_vectorised.clone()
                output_tagger_pos = torch.Tensor([self.PAD_pos]).expand(poss_vectorised.size()).long().clone()
                output_tagger_pos_indcs = torch.LongTensor(list(range(poss_vectorised.size(0))))

                for indx_position in range(poss_vectorised.size()[1]):

                    target_tensor = poss_vectorised[:,indx_position]
                    if self.use_cuda:
                        target_tensor = target_tensor.to('cuda:{}'.format(rank))
                    
                    if self.PAD_pos in target_tensor:
                        indcs_out = (target_tensor != self.PAD_pos).nonzero().reshape(-1)
                        if indcs_out.nelement() == 0:
                            break  
                        target_tensor = target_tensor[indcs_out]
                        if self.model.rnn_attn_pos.rnn_type == 'lstm':
                            decoder_hidden = (decoder_hidden[0][:,indcs_out], decoder_hidden[1][:,indcs_out])
                        else:
                            decoder_hidden = decoder_hidden[:,indcs_out]
                        encoder_output_pos = encoder_output_pos[:,indcs_out]
                        decoder_input = decoder_input[indcs_out]
                        poss_vectorised = poss_vectorised[indcs_out.to('cpu')]
                        output_tagger_pos_indcs = output_tagger_pos_indcs[indcs_out.to('cpu')]
                    if target_tensor.size(0) == 0:
                        break

                    decoder_output, decoder_hidden = self.model.rnn_attn_pos(
                            decoder_input, decoder_hidden, encoder_output_pos)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach()  # detach from history as input
                    output_tagger_pos[output_tagger_pos_indcs,indx_position] = decoder_input.data.cpu().reshape(-1)
                    #print(decoder_output.size(), target_tensor.size())
                    loss_pos += torch.nn.functional.nll_loss(decoder_output, target_tensor, weight=self.pos_label_weight)
                
                
                loss_general_pos += loss_pos.item()
                
            if self.include_morph:

                loss_morph = 0
                decoder_hidden = self.model.init_hidden_morph(morphs_vectorised.size()[0])
                decoder_input = torch.tensor([self.SOS_morph]*morphs_vectorised.size()[0])
                if self.use_cuda:
                    if self.model.rnn_attn_morph.rnn_type == 'lstm':
                        decoder_hidden = (decoder_hidden[0].to('cuda:{}'.format(rank)), decoder_hidden[1].to('cuda:{}'.format(rank)))
                    else:
                        decoder_hidden = decoder_hidden.to('cuda:{}'.format(rank))
                    decoder_input = decoder_input.to('cuda:{}'.format(rank))
                encoder_output_morph = torch.clone(encoder_output)
                encoder_output_morph = encoder_output_morph.unsqueeze(0)
                decoder_hidden = encoder_output_morph#, decoder_hidden[1]
                morphs_vectorised_orig = morphs_vectorised.clone()
                output_tagger_morph = torch.Tensor([self.PAD_morph]).expand(morphs_vectorised.size()).long().clone()
                output_tagger_morph_indcs = torch.LongTensor(list(range(output_tagger_morph.size(0))))

                for indx_position in range(morphs_vectorised.size()[1]):

                    target_tensor = morphs_vectorised[:,indx_position]
                    if self.use_cuda:
                        target_tensor = target_tensor.to('cuda:{}'.format(rank))
                    
                    if self.PAD_morph in target_tensor:
                        indcs_out = (target_tensor != self.PAD_morph).nonzero().reshape(-1)
                        if indcs_out.nelement() == 0:
                            break  
                        target_tensor = target_tensor[indcs_out]
                        if self.model.rnn_attn_morph.rnn_type == 'lstm':
                            decoder_hidden = (decoder_hidden[0][:,indcs_out], decoder_hidden[1][:,indcs_out])
                        else:
                            decoder_hidden = decoder_hidden[:,indcs_out]
                        encoder_output_morph = encoder_output_morph[:,indcs_out]
                        decoder_input = decoder_input[indcs_out]
                        morphs_vectorised = morphs_vectorised[indcs_out.to('cpu')]
                        output_tagger_morph_indcs = output_tagger_morph_indcs[indcs_out.to('cpu')]
                    
                    if target_tensor.size(0) == 0:
                        break

                    # Without teacher forcing: use its own predictions as the next input
                    decoder_output, decoder_hidden = self.model.rnn_attn_morph(
                            decoder_input, decoder_hidden, encoder_output_morph)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach()  # detach from history as input
                    output_tagger_morph[output_tagger_morph_indcs,indx_position] = decoder_input.data.cpu().reshape(-1)
                    #print(decoder_output.size(), target_tensor.size())
                    loss_morph += torch.nn.functional.nll_loss(decoder_output, target_tensor, weight=self.morph_label_weight)
            
                
                loss_general_morph += loss_morph.item()

            '''
            calculating correctness of predictions 
            and
            saving exemplary samples from evaluation for different options lemma-pos-morph
            '''
            
            if self.include_lemma:   
               
                #print(output_lemmatizer)
                #results = self.model.most_similar_lemma(output_lemmatizer).data.cpu().view(-1,self.max_len_lemma).numpy().tolist()
                #results = output_lemmatizer.view(-1,self.max_len_lemma,self.lemma_size).data.cpu().topk(1,dim=2)[1].contiguous().view(-1,self.max_len_lemma).numpy().tolist()
                #quit()
                
                results = [result.replace('<PAD>','') for result in inverse_transform(output_lemmatizer.long().numpy(), self.encoder_lemma)]
              
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
                    
                    myfile = open(os.path.join(self.cellarium, "results_from_eval.txt"), "a", encoding="utf-8")
                    
                    for num, lemma in enumerate(lemmas):
                        
                        if True:# lemma != results[num]:
                        
                            phrases ='{};{}  ---->>>  {}'.format(tokens[num],lemma, results[num])
                        
                            myfile.write('lemmatiser results at epoch no. {}:'.format(nth_iter))
                            myfile.write(phrases)
                            myfile.write('\n')
                    myfile.close()
                        
            if self.include_pos:
               
                pos_original = poss_vectorised_orig.view(-1,self.max_tag_len_seq+1).numpy().tolist()
                #pos_predicted = self.model.most_similar_pos(output_tagger_pos).data.cpu().view(-1,self.max_tag_len_seq).numpy().tolist()
                #pos_predicted = output_tagger_pos.data.cpu().topk(1,dim=1)[1].view(-1,self.max_tag_len_seq).numpy().tolist()
            

                pos_counts, pos_predicted,pos_original, multi_token_result, \
                multi_token_original, multi_tag_result, multi_tag_original \
                = inverse_transform_pos_morph(output_tagger_pos.long().numpy(), pos_original, self.encoder_pos)
            
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
               
                morph_original = morphs_vectorised_orig.view(-1,self.max_tag_len_seq+1).numpy().tolist()
                #morph_predicted = self.model.most_similar_morph(output_tagger_morph).data.cpu().view(-1,self.max_tag_len_seq).numpy().tolist()
                #morph_predicted = output_tagger_morph.data.cpu().topk(1,dim=1)[1].view(-1,self.max_tag_len_seq).numpy().tolist()#.numpy()
                
                morph_counts, morph_predicted, morph_original, multi_token_result, \
                multi_token_original, multi_tag_result, multi_tag_original \
                = inverse_transform_pos_morph(output_tagger_morph.long().numpy(), morph_original, self.encoder_morph)
                
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
    