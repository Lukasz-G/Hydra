import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#from memory_profiler import profile
#from cuda_functional import SRU, SRUCell
#global use_cuda
#use_cuda = torch.cuda.is_available()


class Self_Attn_2D(nn.Module):
    """ Self attention Layer
    from: https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py#L8
    """
    def __init__(self,in_dim):
        super(Self_Attn_2D,self).__init__()
        self.chanel_in = in_dim
        #self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out#,attention

class Self_Attn_3D(nn.Module):
    """ Self attention Layer
    similar as in: https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py#L8
    """
    def __init__(self,in_dim):
        super(Self_Attn_3D,self).__init__()
        self.chanel_in = in_dim
        #self.activation = activation
        
        self.query_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H X D)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height*Depth)
        """
        m_batchsize, C, width, height, depth = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height*depth).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height*depth) # B X C x (*W*H*D)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height*depth) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height,depth)
        
        out = self.gamma*out + x
        return out#,attention



# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size_rnn, hidden_size_output):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size_rnn = hidden_size_rnn
        if self.method == 'general':
            self.attn = nn.Linear(hidden_size_rnn, hidden_size_rnn)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size_rnn, hidden_size_rnn)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size_rnn))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        #print(attn_energies.size())
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, encoder_size, output_size, n_layers=1, dropout=0.1, rnn_type = 'gru'):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn_type = rnn_type

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.embs_size = self.embedding.weight.size(-1)
        #self.squeeze_encoder_outputs = nn.Linear(encoder_size, hidden_size)
        if rnn_type == 'gru':
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        elif rnn_type == 'lstm':
            self.lstm = nn.LSTM(self.embs_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        else:
            raise ValueError(
                f"""Received invalid string for rnn type. Please choose from 'gru' or 'lstm'"""
            )
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)


        self.attn = Attn(attn_model, hidden_size, encoder_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        
        self.gru.flatten_parameters()
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step).reshape(1,-1,self.embs_size)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        #print(embedded.size(), last_hidden.size(), self.lstm)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        #encoder_outputs = self.elu(self.squeeze_encoder_outputs(encoder_outputs))
        # Calculate attention weights from the current GRU output
        #print(rnn_output.size(), encoder_outputs.size())
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        #print(attn_weights.size(), encoder_outputs.size())
        context = attn_weights.bmm(encoder_outputs.permute(1,0,2))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        #rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        #print(rnn_output.size(), context.size())
        concat_input = torch.cat((rnn_output.squeeze(0), context), -1)
        #print()
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        #rnn_output = torch.tanh(concat_output)
        output = self.out(concat_output)
        output = F.log_softmax(output, dim=-1)
        #quit()
        # Return output and final hidden state
        return output.reshape(-1,self.output_size), hidden#, attn_weights
    def init_hidden(self, batch_size):
        if self.rnn_type == 'lstm':
            return torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size)    
        else:
            return torch.zeros(1, batch_size, self.hidden_size)
    



        

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
        
        #self.embedding = False
        #if self.embedding:
        self.embeddingsLettersInput = nn.Embedding(self.letters_size, self.nb_final_kernels_token)
        self.embeddingsLettersInputCL = self.embeddingsLettersInput#nn.Embedding(self.letters_size, self.nb_final_kernels_token)
        self.embeddingsLettersInputCR = self.embeddingsLettersInput#nn.Embedding(self.letters_size, self.nb_final_kernels_token)
        self.embeddingsLettersOutput = nn.Embedding(self.lemma_size, self.nb_final_kernels_token)
        self.embeddingsOfPoS = nn.Embedding(self.pos_size, self.nb_final_kernels_token)
        self.embeddingsMorph = nn.Embedding(self.morph_size, self.nb_final_kernels_token)


        
        #self.letters_size = self.nb_final_kernels_token
        
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
      
            
        dropout_value_conv = 0.0
        dropout_value_dense = 0.0
        kernel_size_01 = (self.max_len_tok+4-(3-1)-1)//2 + 1
        kernel_size_02 = (kernel_size_01+2-(3-1)-1)//2 + 1
        #print(kernel_size_02, 'kernel_size_02')
        self.token = nn.Sequential(
            #nn.Dropout(dropout_value_conv),
            nn.Conv2d(1, self.nb_kernels_token, 
            kernel_size=(3,self.nb_final_kernels_token), stride=(2,1), padding=(2,0)),
            #Self_Attn_2D(self.nb_kernels_token),
            nn.Dropout2d(dropout_value_conv),
            nn.MaxPool2d((3, 1), stride=(2, 1), padding=(1,0)),
            ##nn.BatchNorm2d(self.nb_kernels_token),
            nn.ELU(),
            #nn.Conv2d(self.nb_kernels_token, self.nb_kernels_token, 
            #          kernel_size=(3,1), stride=(2,1), padding=(2,0)),
            #nn.BatchNorm2d(self.nb_kernels_token),
            #nn.Dropout2d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv2d(self.nb_kernels_token, self.nb_final_kernels_token, 
            #          kernel_size=(kernel_size_01,1), stride=(1,1), padding=(0,0)),
            #nn.ELU()
            )
        
        self.token_lin = nn.Linear(kernel_size_02*self.nb_final_kernels_token, nb_final_kernels_token)
        
        self.minicontext_L = nn.Sequential(
            #nn.Dropout(dropout_value_conv),
            nn.Conv2d(1, self.nb_kernels_token, 
            kernel_size=(3,self.nb_final_kernels_token), stride=(2,1), padding=(2,0)),
            #Self_Attn_2D(self.nb_kernels_token),
            nn.Dropout2d(dropout_value_conv),
            nn.MaxPool2d((3, 1), stride=(2, 1), padding=(1,0)),
            ##nn.BatchNorm2d(self.nb_kernels_token),
            nn.ELU(),
            #nn.Conv2d(self.nb_kernels_token, self.nb_kernels_token, 
            #          kernel_size=(3,1), stride=(2,1), padding=(2,0)),
            #nn.BatchNorm2d(self.nb_kernels_token),
            #nn.Dropout2d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv2d(self.nb_kernels_token, self.nb_final_kernels_token, 
            #          kernel_size=(kernel_size_01,1), stride=(1,1), padding=(0,0)),
            #nn.ELU()
               )
        
        self.minicontext_L_lin = nn.Linear(kernel_size_02*self.nb_final_kernels_token, nb_final_kernels_token)
        
        self.minicontext_R = nn.Sequential(
            #nn.Dropout(dropout_value_conv),
            nn.Conv2d(1, self.nb_kernels_token, 
            kernel_size=(3,self.nb_final_kernels_token), stride=(2,1), padding=(2,0)),
            #Self_Attn_2D(self.nb_kernels_token),
            nn.Dropout2d(dropout_value_conv),
            nn.MaxPool2d((3, 1), stride=(2, 1), padding=(1,0)),
            ##nn.BatchNorm2d(self.nb_kernels_token),
            nn.ELU(),
            #nn.Conv2d(self.nb_kernels_token, self.nb_kernels_token, 
            #          kernel_size=(3,1), stride=(2,1), padding=(2,0)),
            #nn.BatchNorm2d(self.nb_kernels_token),
            #nn.Dropout2d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv2d(self.nb_kernels_token, self.nb_final_kernels_token, 
            #          kernel_size=(kernel_size_01,1), stride=(1,1), padding=(0,0)),
            #nn.ELU()
            )
        
        self.minicontext_R_lin = nn.Linear(kernel_size_02*self.nb_final_kernels_token, nb_final_kernels_token)
        
       
        
        kernel_size_1a = (self.max_len_tok+4-(3-1)-1)//2 + 1
        kernel_size_1b = (kernel_size_1a+2-(3-1)-1)//1 + 1
        #kernel_size_2 = (self.context_size+4-(3-1)-1)//2 + 1
        #self.context_size
        kernel_size_2a = (self.context_size+4-(3-1)-1)//2 + 1
        kernel_size_2b = (kernel_size_2a+2-(3-1)-1)//1 + 1
        
        self.context_L = nn.Sequential(
            nn.Dropout(dropout_value_conv),
           nn.Conv3d(1, self.nb_kernels_c_L, kernel_size=(3, 3, self.nb_final_kernels_token), stride=(2,2,1), padding=(2,2,0)),
            ##nn.BatchNorm3d(self.nb_kernels_c_L),
            #Self_Attn_3D(self.nb_kernels_c_L),
            nn.MaxPool3d((3, 3, 1), stride=(1, 1, 1), padding=(1,1,0)),
            nn.Dropout3d(dropout_value_conv),
            nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_L, self.nb_kernels_c_L, kernel_size=(1,3,1), stride=(1,2,1), padding=(0,2,0)),
            ##nn.BatchNorm3d(self.nb_kernels_c_L),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_L, self.nb_final_kernels_c_L, kernel_size=(1,kernel_size_1b,1), stride=(1,1,1), padding=(0,0,0)),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_L, self.nb_final_kernels_c_L, kernel_size=(3,1,1), stride=(2,1,1), padding=(2,0,0)),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_L, self.nb_final_kernels_c_L, kernel_size=(kernel_size_2,1,1), stride=(1,1,1), padding=(0,0,0)),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            )
        self.context_L_lin = nn.Linear(kernel_size_1b*kernel_size_2b*self.nb_final_kernels_token, nb_final_kernels_token)
        
        self.context_R = nn.Sequential(
            nn.Dropout(dropout_value_conv),
           nn.Conv3d(1, self.nb_kernels_c_R, kernel_size=(3, 3, self.nb_final_kernels_token), stride=(2,2,1), padding=(2,2,0)),
            ##nn.BatchNorm3d(self.nb_kernels_c_L),
            #Self_Attn_3D(self.nb_kernels_c_R),
            nn.MaxPool3d((3, 3, 1), stride=(1, 1, 1), padding=(1,1,0)),
            nn.Dropout3d(dropout_value_conv),
            nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_R, self.nb_kernels_c_R, kernel_size=(1,3,1), stride=(1,2,1), padding=(0,2,0)),
            ##nn.BatchNorm3d(self.nb_kernels_c_L),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_R, self.nb_final_kernels_c_R, kernel_size=(1,kernel_size_1b,1), stride=(1,1,1), padding=(0,0,0)),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_R, self.nb_final_kernels_c_R, kernel_size=(3,1,1), stride=(2,1,1), padding=(2,0,0)),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_R, self.nb_final_kernels_c_R, kernel_size=(kernel_size_2,1,1), stride=(1,1,1), padding=(0,0,0)),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            )

        self.context_R_lin = nn.Linear(kernel_size_1b*kernel_size_2b*self.nb_final_kernels_token, nb_final_kernels_token)
        
        self.input_size_together_1 = self.nb_final_kernels_token+self.nb_final_kernels_minic_L+self.nb_final_kernels_minic_R
        self.input_size_together_2 = self.nb_final_kernels_token + self.nb_final_kernels_c_L + self.nb_final_kernels_c_R
        
        self.together1 = nn.Linear(self.input_size_together_1,
                                   self.nb_final_kernels_token)
        
        self.together2 = nn.Linear(self.nb_final_kernels_token*5,self.nb_final_kernels_token)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dense = nn.Linear(self.input_size_together_2,self.input_size_together_2, bias=False)

        vector_size_after_convs = (kernel_size_1b*kernel_size_2b*self.nb_final_kernels_token)*2 + kernel_size_02*self.nb_final_kernels_token
        self.rnn_attn_lemmas = LuongAttnDecoderRNN(attn_model="general", embedding=self.embeddingsLettersOutput, hidden_size=self.nb_final_kernels_token, \
            encoder_size=vector_size_after_convs, output_size=self.lemma_size, n_layers=1, dropout=0.0)

        self.rnn_attn_pos = LuongAttnDecoderRNN(attn_model="general", embedding=self.embeddingsOfPoS, hidden_size=self.nb_final_kernels_token, \
            encoder_size=vector_size_after_convs, output_size=self.pos_size, n_layers=1, dropout=0.0)

        self.rnn_attn_morph = LuongAttnDecoderRNN(attn_model="general", embedding=self.embeddingsMorph, hidden_size=self.nb_final_kernels_token, \
            encoder_size=vector_size_after_convs, output_size=self.morph_size, n_layers=1, dropout=0.0)

        
       
        
        

      
        #for lemma
        if include_lemma:
            #depth_dense_layer_lemma = 2
            #self.layer_lemma =  nn.ModuleList(\
            #[nn.Sequential(nn.Dropout(dropout_value_dense), nn.ELU(), nn.Linear(self.lemma_size, self.lemma_size))\
            # if x != depth_dense_layer_lemma-0\
            # else\
            # nn.Sequential(nn.ELU(), nn.Linear(self.lemma_size, self.lemma_size))
            # for x in range(depth_dense_layer_lemma)])#nn.Linear(self.nb_final_kernels, self.nb_final_kernels)
            #self.final_layer_lemma = nn.Linear(self.nb_final_kernels_token, self.lemma_size*self.max_len_lemma)
            pass
        #for pos
        if include_pos:
            #depth_dense_layer_pos = 2
            #self.layer_pos = nn.ModuleList(\
            #[nn.Sequential(nn.Dropout(dropout_value_dense), nn.ELU(), nn.Linear(self.pos_size, self.pos_size))\
            #if x != depth_dense_layer_pos-0\
            #else\
            #nn.Sequential(nn.ELU(), nn.Linear(self.pos_size, self.pos_size))
            #for x in range(depth_dense_layer_pos)])#nn.Linear(self.nb_final_kernels, self.nb_final_kernels)
            self.final_layer_pos = nn.Linear(self.nb_final_kernels_token, self.pos_size)
        #for morph
        if include_morph:
            #depth_dense_layer_morph = 2
            #self.layer_morph = nn.ModuleList(\
            #[nn.Sequential(nn.Dropout(dropout_value_dense), nn.ELU(),nn.Linear(self.morph_size, self.morph_size))\
            #if x != depth_dense_layer_morph-0\
            #else\
            #nn.Sequential(nn.ELU(), nn.Linear(self.morph_size, self.morph_size))
            #for x in range(depth_dense_layer_morph)])#nn.Linear(self.nb_final_kernels, self.nb_final_kernels)
            self.final_layer_morph = nn.Linear(self.nb_final_kernels_token, self.morph_size)
    

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

    def init_hidden_lemmas(self, batch_size):

        return self.rnn_attn_lemmas.init_hidden(batch_size)         
    
    def init_hidden_pos(self, batch_size):

        return self.rnn_attn_pos.init_hidden(batch_size)
    
    def init_hidden_morph(self, batch_size):

        return self.rnn_attn_morph.init_hidden(batch_size)
    
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
    
    
    
    def encode(self, tokens=None,
                context_left=None, 
                context_right=None,
                ):
            
            minicontext_L = context_left[-1,:,:].unsqueeze(0)
            minicontext_R = context_right[0,:,:].unsqueeze(0)
            #print(tokens[1])
            #if not self.embedding:
            #    tokens = self.make_one_hot(tokens, C = self.letters_size)
            #    context_left = self.make_one_hot(context_left, C = self.letters_size)
            #   context_right = self.make_one_hot(context_right, C = self.letters_size)
            #    minicontext_L = self.make_one_hot(minicontext_L, C = self.letters_size)
            #    minicontext_R = self.make_one_hot(minicontext_R, C = self.letters_size)
            #else:
            tokens = self.embeddingsLettersInput(tokens).transpose(1,0)
            context_left = self.embeddingsLettersInputCL(context_left).transpose(1,0).unsqueeze(1)
            context_right = self.embeddingsLettersInputCR(context_right).transpose(1,0).unsqueeze(1)
            minicontext_L = self.embeddingsLettersInput(minicontext_L).transpose(1,0)#.squeeze()
            minicontext_R = self.embeddingsLettersInput(minicontext_R).transpose(1,0)#.squeeze()
            
            #print(tokens.size(), context_left.size())
            #quit()
            #tokens = tokens.reshape(self.max_len_tok,-1,self.nb_final_kernels_token).transpose(1,0).unsqueeze(1)
            #context_left = context_left.reshape(-1, 1,self.context_size,self.max_len_tok,self.nb_final_kernels_token) 
            #context_right = context_right.reshape(-1, 1,self.context_size,self.max_len_tok,self.nb_final_kernels_token)
            
            #minicontext_L = minicontext_L.reshape(self.max_len_tok,-1,self.nb_final_kernels_token).transpose(1,0).unsqueeze(1)
            #minicontext_R = minicontext_R.reshape(self.max_len_tok,-1,self.nb_final_kernels_token).transpose(1,0).unsqueeze(1)

            current_batch_size = tokens.size(0)
            tokens = self.token(tokens).reshape(current_batch_size, -1)
            
            #print(tokens.size())
            tokens = self.elu(self.token_lin(tokens))
            #print(tokens.size())
            
            #quit()
            
            minicontext_L = self.minicontext_L(minicontext_L).reshape(current_batch_size, -1)
            minicontext_R = self.minicontext_R(minicontext_R).reshape(current_batch_size, -1)
            minicontext_L = self.elu(self.minicontext_L_lin(minicontext_L))
            minicontext_R = self.elu(self.minicontext_R_lin(minicontext_R))
           
            #token_context = torch.cat([minicontext_L, tokens, minicontext_R],1).reshape(-1, self.input_size_together_1)#view(-1,1,1,self.nb_final_kernels*3)#
            
            
            #token_context = self.together1(token_context)
            #token_context = self.elu(token_context)
        
        
            context_left = self.context_L(context_left).reshape(current_batch_size, -1)#.reshape(-1, self.nb_final_kernels_c_L)
            context_left = self.elu(self.context_L_lin(context_left))
            #print(context_left.size())

            #quit()
            context_right = self.context_R(context_right).reshape(current_batch_size, -1)#.reshape(-1, self.nb_final_kernels_c_R)
            context_right = self.elu(self.context_R_lin(context_right))
            mix = torch.cat([tokens, minicontext_L, minicontext_R, context_left, context_right],1)#.reshape(-1, self.nb_final_kernels_token*3)#view(-1,1,1,self.nb_final_kernels*3) view(-1, self.nb_final_kernels*3)
          
            #attention for dense layer
            #attn_vector = self.softmax(self.tanh(self.attn_dense(mix)))
            #mix = mix * attn_vector
            mix = self.together2(mix)
            mix = self.elu(mix)
            #mix = context_left + tokens + context_right# + minicontext_L + minicontext_R

            return mix
          
            #output_lemmatizer, output_tagger_pos, output_tagger_morph = None, None, None
            
            
            #if self.include_lemma:
            #    mix_for_lemma = mix.view(-1, self.nb_final_kernels_token, 1, 1)
            #    mix_for_lemma = self.lemmas_deconv(mix_for_lemma).view(-1, self.max_len_lemma, self.lemma_size)
            #    output_lemmatizer = mix_for_lemma.view(-1, self.lemma_size)
                #for l in self.layer_lemma:
                #   output_lemmatizer = l(output_lemmatizer)
            #    output_lemmatizer = F.log_softmax(output_lemmatizer,dim=1)
                
            #if self.include_pos:    
            #    mix_for_pos = mix.view(-1, self.nb_final_kernels_token, 1, 1)
            #    mix_for_pos = self.pos_deconv(mix_for_pos).view(-1, self.max_tag_len_seq, self.pos_size)
            #    output_tagger_pos = mix_for_pos.view(-1, self.pos_size)
                #for l in self.layer_pos:
                #    output_tagger_pos = l(output_tagger_pos)
            #    output_tagger_pos = F.log_softmax(output_tagger_pos,dim=1)
            
            #if self.include_morph:
            #    mix_for_morph = mix.view(-1, self.nb_final_kernels_token, 1, 1)
            #    mix_for_morph = self.morph_deconv(mix_for_morph).view(-1, self.max_tag_len_seq, self.morph_size) 
            #    output_tagger_morph = mix_for_morph.view(-1, self.morph_size)
                #for l in self.layer_morph:
                #    output_tagger_morph = l(output_tagger_morph)
            #    output_tagger_morph = F.log_softmax(output_tagger_morph,dim=1)
           
            
            #return output_lemmatizer, output_tagger_pos, output_tagger_morph, lemmas

 