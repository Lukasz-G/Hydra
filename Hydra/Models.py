import socket
from unicodedata import bidirectional
hostname = socket.gethostname()

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from memory_profiler import profile
#from cuda_functional import SRU, SRUCell
#global use_cuda
#use_cuda = torch.cuda.is_available()



        

'''model '''
class ModelConvDecon(nn.Module):
    def __init__(self, embedder_size=0, hidden_size_model = 0,
                 include_lemma = None, include_pos = None, include_morph = None,
                 max_len_tok = 0, letters_size=0, lemma_size =0, pos_size=0, morph_size=0,
                 letters_pad = 0,
                 n_layers=0, context_size=0, max_len_lemma=0, max_tag_len_seq=0, use_cuda=False,
                 nb_kernels_token = 1, nb_final_kernels_token = 1, 
                 nb_kernels_minic_L = 1, nb_final_kernels_minic_L = 1,
                 nb_kernels_minic_R = 1, nb_final_kernels_minic_R = 1,
                 nb_kernels_c_L = 1, nb_final_kernels_c_L = 1,
                 nb_kernels_c_R = 1, nb_final_kernels_c_R = 1
                 ):
        super(ModelConvDecon, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size_model = hidden_size_model
        self.max_len_tok = max_len_tok
        
        self.embedder_size = embedder_size
        self.lemma_size = lemma_size
        self.pos_size = pos_size
        self.morph_size = morph_size
        self.letters_size = letters_size
        self.context_size = context_size
        self.max_len_lemma = max_len_lemma
        self.max_tag_len_seq = max_tag_len_seq
        self.letters_pad = letters_pad
        
        self.use_cuda = use_cuda
        
        self.nb_kernels_token = nb_kernels_token
        self.nb_final_kernels_token = nb_final_kernels_token
        
        self.nb_kernels_minic_L = nb_kernels_minic_L
        self.nb_final_kernels_minic_L = nb_final_kernels_minic_L
        
        self.nb_kernels_minic_R = nb_kernels_minic_R
        self.nb_final_kernels_minic_R = nb_final_kernels_minic_R
        
        self.nb_kernels_c_L = nb_kernels_c_L
        self.nb_final_kernels_c_L = nb_final_kernels_c_L
        
        self.nb_kernels_c_R = nb_kernels_c_R
        self.nb_final_kernels_c_R = nb_final_kernels_c_R
        
        self.include_lemma, self.include_pos, self.include_morph = include_lemma, include_pos, include_morph
        
        self.embedding = False
        if self.embedding:
            self.embeddings_input = nn.Embedding(self.letters_size, 32)
            self.letters_size = 32
        
        self.elu = nn.ELU()
      
            
        dropout_value_conv = 0.0
        dropout_value_dense = 0.0
        kernel_size_01 = (self.max_len_tok+4-(3-1)-1)//2 + 1
        kernel_size_02 = (kernel_size_01+4-(3-1)-1)//2 + 1
        
        self.token = nn.Sequential(
            nn.Dropout(dropout_value_conv),
            nn.Conv2d(1, self.nb_kernels_token, 
            kernel_size=(3,self.letters_size), stride=(2,1), padding=(2,0)),
            nn.Dropout2d(dropout_value_conv),
            nn.BatchNorm2d(self.nb_kernels_token),
            nn.ELU(),
            nn.Conv2d(self.nb_kernels_token, self.nb_kernels_token, 
                      kernel_size=(3,1), stride=(2,1), padding=(2,0)),
            nn.BatchNorm2d(self.nb_kernels_token),
            nn.Dropout2d(dropout_value_conv),
            nn.ELU(),
            nn.Conv2d(self.nb_kernels_token, self.nb_final_kernels_token, 
                      kernel_size=(kernel_size_02,1), stride=(1,1), padding=(0,0)),
            )
        
       
        
        self.minicontext_L = nn.Sequential(
            nn.Dropout(dropout_value_conv),
            nn.Conv2d(1, self.nb_kernels_minic_L, 
            kernel_size=(3,self.letters_size), stride=(2,1), padding=(2,0)),
            nn.Dropout2d(dropout_value_conv),
            nn.BatchNorm2d(self.nb_kernels_minic_L),
            nn.ELU(),
            nn.Conv2d(self.nb_kernels_minic_L, self.nb_kernels_minic_L, 
                      kernel_size=(3,1), stride=(2,1), padding=(2,0)),
            nn.BatchNorm2d(self.nb_kernels_minic_L),
            nn.Dropout2d(dropout_value_conv),
            nn.ELU(),
            nn.Conv2d(self.nb_kernels_minic_L, self.nb_final_kernels_minic_L, 
                      kernel_size=(kernel_size_02,1), stride=(1,1), padding=(0,0)),
               )
        
        self.minicontext_R = nn.Sequential(
            nn.Dropout(dropout_value_conv),
            nn.Conv2d(1, self.nb_kernels_minic_R, 
            kernel_size=(3,self.letters_size), stride=(2,1), padding=(2,0)),
            nn.Dropout2d(dropout_value_conv),
            nn.BatchNorm2d(self.nb_kernels_minic_R),
            nn.ELU(),
            nn.Conv2d(self.nb_kernels_minic_R, self.nb_kernels_minic_R, 
                      kernel_size=(3,1), stride=(2,1), padding=(2,0)),
            nn.BatchNorm2d(self.nb_kernels_minic_R),
            nn.Dropout2d(dropout_value_conv),
            nn.ELU(),
            nn.Conv2d(self.nb_kernels_minic_R, self.nb_final_kernels_minic_R, 
                      kernel_size=(kernel_size_02,1), stride=(1,1), padding=(0,0)),
          
            )
        
       
        
        kernel_size_1a = (self.max_len_tok+4-(3-1)-1)//2 + 1
        kernel_size_1b = (kernel_size_1a+4-(3-1)-1)//2 + 1
        kernel_size_2 = (self.context_size+4-(3-1)-1)//2 + 1
        
        
        
        self.context_L = nn.Sequential(
            nn.Dropout(dropout_value_conv),
           nn.Conv3d(1, self.nb_kernels_c_L, kernel_size=(1, 3, self.letters_size), stride=(1,2,1), padding=(0,2,0)),
            nn.BatchNorm3d(self.nb_kernels_c_L),
            nn.Dropout3d(dropout_value_conv),
            nn.ELU(),
            nn.Conv3d(self.nb_kernels_c_L, self.nb_kernels_c_L, kernel_size=(3,3,1), stride=(2,2,1), padding=(2,2,0)),
            nn.BatchNorm3d(self.nb_kernels_c_L),
            nn.Dropout3d(dropout_value_conv),
            nn.ELU(),
            nn.Conv3d(self.nb_kernels_c_L, self.nb_final_kernels_c_L, kernel_size=(kernel_size_2,kernel_size_1b,1))
            )
        
        self.context_R = nn.Sequential(
                 nn.Dropout(dropout_value_conv),
           nn.Conv3d(1, self.nb_kernels_c_R, kernel_size=(1, 3, self.letters_size), stride=(1,2,1), padding=(0,2,0)),
            nn.BatchNorm3d(self.nb_kernels_c_R),
            nn.Dropout3d(dropout_value_conv),
            nn.ELU(),
            nn.Conv3d(self.nb_kernels_c_R, self.nb_kernels_c_R, kernel_size=(3,3,1), stride=(2,2,1), padding=(2,2,0)),
            nn.BatchNorm3d(self.nb_kernels_c_R),
            nn.Dropout3d(dropout_value_conv),
            nn.ELU(),
            nn.Conv3d(self.nb_kernels_c_R, self.nb_final_kernels_c_R, kernel_size=(kernel_size_2,kernel_size_1b,1))
            )

        
        self.input_size_together_1 = self.nb_final_kernels_token+self.nb_final_kernels_minic_L+self.nb_final_kernels_minic_R
        self.input_size_together_2 = self.nb_final_kernels_token + self.nb_final_kernels_c_L + self.nb_final_kernels_c_R
        
        self.together1 = nn.Linear(self.input_size_together_1,
                                   self.nb_final_kernels_token)
        
        self.together2 = nn.Linear(self.input_size_together_2,self.nb_final_kernels_token)
       
        
        
        kernel_size_deconv1a = (1-1)*2 - 2*0 + 3 + 0
        kernel_size_deconv2a = (kernel_size_deconv1a-1)*2 - 2*0 + 3 + 0
        kernel_size_deconv1b = (1-1)*2 - 2*0 + 3 + 0
        kernel_size_deconv2b = (kernel_size_deconv1b-1)*2 - 2*0 + 3 + 0
        kernel_size_deconv3a = self.max_len_lemma - kernel_size_deconv2a + 1
        kernel_size_deconv3b = self.lemma_size - kernel_size_deconv2b + 1
        
        
        self.lemmas_deconv = nn.Sequential(
            nn.Dropout(dropout_value_conv),
            nn.ELU(),
            nn.ConvTranspose2d(self.nb_final_kernels_token, self.nb_final_kernels_token, 
            kernel_size=(3, 3), stride=(2,2), padding=(0,0)),
            nn.BatchNorm2d(self.nb_final_kernels_token),
            nn.Dropout2d(dropout_value_conv),
            nn.ELU(),
            nn.ConvTranspose2d(self.nb_final_kernels_token, self.nb_final_kernels_token, 
                      kernel_size=(3, 3), stride=(2,2), padding=(0,0)),
            nn.BatchNorm2d(self.nb_final_kernels_token),
            nn.Dropout2d(dropout_value_conv),
            nn.ELU(),
            nn.ConvTranspose2d(self.nb_final_kernels_token, 1, 
                      kernel_size=(kernel_size_deconv3a, kernel_size_deconv3b), stride=(1,1), padding=(0,0)),
        
            )
        
        kernel_size_deconv1a = (1-1)*2 - 2*0 + 3 + 0
        kernel_size_deconv1b = (1-1)*2 - 2*0 + 3 + 0
        kernel_size_deconv2a = self.max_tag_len_seq - kernel_size_deconv1a + 1# + 2 - 2#(1-1)*2 - 2*0 + 1 + 0
        kernel_size_deconv2b = self.pos_size - kernel_size_deconv1b + 1# + 2 - 2
     
        
        
        self.pos_deconv = nn.Sequential(
            nn.Dropout(dropout_value_conv),
            nn.ELU(),
            nn.ConvTranspose2d(self.nb_final_kernels_token, self.nb_final_kernels_token, 
            kernel_size=(3, 3), stride=(2,2), padding=(0,0)),
            nn.Dropout2d(dropout_value_conv),
            nn.ELU(),
            nn.BatchNorm2d(self.nb_final_kernels_token),
            nn.ConvTranspose2d(self.nb_final_kernels_token, 1, 
                      kernel_size=(kernel_size_deconv2a, kernel_size_deconv2b), stride=(1,1), padding=(0,0)),
    
            )
        
        kernel_size_deconv1a = (1-1)*2 - 2*0 + 3 + 0
        kernel_size_deconv1b = (1-1)*2 - 2*0 + 3 + 0
        kernel_size_deconv2a = self.max_tag_len_seq - kernel_size_deconv1a + 1# + 2 - 2#(1-1)*2 - 2*0 + 1 + 0
        kernel_size_deconv2b = self.morph_size - kernel_size_deconv1b + 1# + 2 - 2
        
     
        
        self.morph_deconv = nn.Sequential(
            nn.Dropout(dropout_value_conv),
            nn.ELU(),
            nn.ConvTranspose2d(self.nb_final_kernels_token, self.nb_final_kernels_token, 
            kernel_size=(3, 3), stride=(2,2), padding=(0,0)),
            nn.Dropout2d(dropout_value_conv),
            nn.ELU(),
            nn.BatchNorm2d(self.nb_final_kernels_token),
        
            nn.ConvTranspose2d(self.nb_final_kernels_token, 1, 
                      kernel_size=(kernel_size_deconv2a, kernel_size_deconv2b), stride=(1,1), padding=(0,0)),
    
            )

      
        #for lemma
        if include_lemma:
            depth_dense_layer_lemma = 2
            self.layer_lemma =  nn.ModuleList(\
            [nn.Sequential(nn.Dropout(dropout_value_dense), nn.ELU(), nn.Linear(self.lemma_size, self.lemma_size))\
             if x != depth_dense_layer_lemma-0\
             else\
             nn.Sequential(nn.ELU(), nn.Linear(self.lemma_size, self.lemma_size))
             for x in range(depth_dense_layer_lemma)])#nn.Linear(self.nb_final_kernels, self.nb_final_kernels)
            #self.final_layer_lemma = nn.Linear(self.nb_final_kernels_token, self.lemma_size*self.max_len_lemma)
        #for pos
        if include_pos:
            depth_dense_layer_pos = 2
            self.layer_pos = nn.ModuleList(\
            [nn.Sequential(nn.Dropout(dropout_value_dense), nn.ELU(), nn.Linear(self.pos_size, self.pos_size))\
            if x != depth_dense_layer_pos-0\
            else\
            nn.Sequential(nn.ELU(), nn.Linear(self.pos_size, self.pos_size))
            for x in range(depth_dense_layer_pos)])#nn.Linear(self.nb_final_kernels, self.nb_final_kernels)
            
            
            
            #self.final_layer_pos = nn.Linear(self.nb_final_kernels_token, self.pos_size*max_tag_len_seq)
        #for morph
        if include_morph:
            depth_dense_layer_morph = 2
            self.layer_morph = nn.ModuleList(\
            [nn.Sequential(nn.Dropout(dropout_value_dense), nn.ELU(),nn.Linear(self.morph_size, self.morph_size))\
            if x != depth_dense_layer_morph-0\
            else\
            nn.Sequential(nn.ELU(), nn.Linear(self.morph_size, self.morph_size))
            for x in range(depth_dense_layer_morph)])#nn.Linear(self.nb_final_kernels, self.nb_final_kernels)
            #self.final_layer_morph = nn.Linear(self.nb_final_kernels_token, self.morph_size*max_tag_len_seq)
    

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n)**0.5)
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n)**0.5)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n)**0.5)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, (2.0 / n)**0.5)
                
    
    def make_one_hot(self, labels, C=2):
        '''
        shamelessly stolen from: http://jacobkimmel.github.io/pytorch_onehot/
        
        '''
       
        if labels.size(2) == 1:
            one_hot = torch.FloatTensor(labels.size(0), labels.size(1), C).zero_()
            target = one_hot.scatter_(2, labels, 1)
        elif labels.size(2) > 1:
            labels = labels.unsqueeze(3)
            one_hot = torch.FloatTensor(labels.size(0), labels.size(1), labels.size(2),C).zero_()
            target = one_hot.scatter_(3, labels, 1)
            
        return target
    
    def make_variables(self, tensor):
        if self.training == False:    
            requires_grad = False
        else:
            requires_grad = True
        
        var = Variable(tensor)
        if self.use_cuda:
            var = var.cuda()
        
        return var 
    
    def list_embedder(self,list_to_emb, embedder):
        
        new_list = []
        for var in list_to_emb:
            var = self.make_variables(var)
            emb_letters = embedder(input=var).contiguous()
            new_list.append(emb_letters)
        
        return new_list
    
    
    
    def forward(self, tokens=None,
                context_left=None, 
                context_right=None,
                lemmas=None
                ):
            
            minicontext_L = context_left[-1,:,:].unsqueeze(0)
            minicontext_R = context_right[0,:,:].unsqueeze(0)
            
            if not self.embedding:
                tokens = self.make_one_hot(tokens, C = self.letters_size)
                context_left = self.make_one_hot(context_left, C = self.letters_size)
                context_right = self.make_one_hot(context_right, C = self.letters_size)
                minicontext_L = self.make_one_hot(minicontext_L, C = self.letters_size)
                minicontext_R = self.make_one_hot(minicontext_R, C = self.letters_size)
            else:
                tokens = self.embeddings_input(tokens)
                context_left = self.embeddings_input(context_left).squeeze()
                context_right = self.embeddings_input(context_right).squeeze()
                minicontext_L = self.embeddings_input(minicontext_L).squeeze()
                minicontext_R = self.embeddings_input(minicontext_R).squeeze()
            
            tokens = tokens.contiguous().view(self.max_len_tok,-1,self.letters_size).transpose(1,0).unsqueeze(1)
            context_left = context_left.contiguous().view(-1, 1,self.context_size,self.max_len_tok,self.letters_size) 
            context_right = context_right.contiguous().view(-1, 1,self.context_size,self.max_len_tok,self.letters_size)
            
            minicontext_L = minicontext_L.contiguous().view(self.max_len_tok,-1,self.letters_size).transpose(1,0).unsqueeze(1)
            minicontext_R = minicontext_R.contiguous().view(self.max_len_tok,-1,self.letters_size).transpose(1,0).unsqueeze(1)
        
           
            tokens = self.token(tokens).view(-1, self.nb_final_kernels_token)
            minicontext_L = self.minicontext_L(minicontext_L).view(-1, self.nb_final_kernels_minic_L)
            minicontext_R = self.minicontext_R(minicontext_R).view(-1, self.nb_final_kernels_minic_R)
        
           
            token_context = torch.cat([minicontext_L, tokens, minicontext_R],1).view(-1, self.input_size_together_1)#view(-1,1,1,self.nb_final_kernels*3)#
            
            
            token_context = self.elu(token_context)
            token_context = self.together1(token_context)
        
        
        
            context_left = self.context_L(context_left).view(-1, self.nb_final_kernels_c_L)
            context_right = self.context_R(context_right).view(-1, self.nb_final_kernels_c_R)
    
            mix = torch.cat([context_left, token_context, context_right],1).view(-1, self.input_size_together_2)#view(-1,1,1,self.nb_final_kernels*3) view(-1, self.nb_final_kernels*3)
          
            mix = self.elu(mix)
            mix = self.together2(mix)
          
            output_lemmatizer, output_tagger_pos, output_tagger_morph = None, None, None
            
            
            if self.include_lemma:
                mix_for_lemma = mix.view(-1, self.nb_final_kernels_token, 1, 1)
                mix_for_lemma = self.lemmas_deconv(mix_for_lemma).view(-1, self.max_len_lemma, self.lemma_size)
                output_lemmatizer = mix_for_lemma.view(-1, self.lemma_size)
                for l in self.layer_lemma:
                    output_lemmatizer = l(output_lemmatizer)
                output_lemmatizer = F.log_softmax(output_lemmatizer,dim=1)
                
            if self.include_pos:    
                mix_for_pos = mix.view(-1, self.nb_final_kernels_token, 1, 1)
                mix_for_pos = self.pos_deconv(mix_for_pos).view(-1, self.max_tag_len_seq, self.pos_size)
                output_tagger_pos = mix_for_pos.view(-1, self.pos_size)
                for l in self.layer_pos:
                    output_tagger_pos = l(output_tagger_pos)
                output_tagger_pos = F.log_softmax(output_tagger_pos,dim=1)
            
            if self.include_morph:
                mix_for_morph = mix.view(-1, self.nb_final_kernels_token, 1, 1)
                mix_for_morph = self.morph_deconv(mix_for_morph).view(-1, self.max_tag_len_seq, self.morph_size) 
                output_tagger_morph = mix_for_morph.view(-1, self.morph_size)
                for l in self.layer_morph:
                    output_tagger_morph = l(output_tagger_morph)
                output_tagger_morph = F.log_softmax(output_tagger_morph,dim=1)
           
            
            return output_lemmatizer, output_tagger_pos, output_tagger_morph, lemmas

 