import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
#from memory_profiler import profile
#from cuda_functional import SRU, SRUCell
#global use_cuda
#use_cuda = torch.cuda.is_available()

from torch.nn.utils import weight_norm
#from torchdyn.core import NeuralDE



class sequentialMultiInput(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            #print(type(inputs))
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True, device="cuda"):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.Tensor(1).float().to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

class Node(nn.Module):
    def __init__(self, hidden_size):
        super(Node, self).__init__()
        self.hidden_size = hidden_size
        # 1 hidden layer NODE
        self.f_node = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            )

    def forward(self, x, tspan):
        
        #permute
        #print(tspan.size())
        #quit()
        tspan_trimmed = torch.linspace(0,1,x.shape[-1], device=x.device, requires_grad=True).reshape(1,-1, 1)
        #tspan = tspan.reshape(1,-1, 1)
        
        #tspan_trimmed = tspan - tspan[:,0].unsqueeze(1).expand(-1,tspan.size(1))
        #print(tspan_trimmed)
        #quit()
        x = torch.permute(x, (0,2,1))
        #tspan_trimmed = tspan_trimmed.unsqueeze(2).expand(-1,-1,x.size(2))
        tspan_trimmed = tspan_trimmed.expand(x.size(0),-1,x.size(2))
        #print(tspan_trimmed)
        #quit()
        #tspan = tspan.expand(x.size(0),-1,x.size(2))
        #solve
        x = self.solve_fixed(x, tspan_trimmed)
        #permute back
        x = torch.permute(x, (0,2,1))

        return x
        #return
    
    def solve_fixed(self, x, ts):
        for i in range(3):  # 3 unfolds
            x = self.euler(x, ts * (1.0 / 3))
        return x
    
    def euler(self, y, delta_t):
        dy = self.f_node(y)
        #print(dy.size(), delta_t.size())
        return y + delta_t * dy

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class CutOut1d(nn.Module):
    def __init__(self, cut_size):
        super(CutOut1d, self).__init__()
        self.cut_size = cut_size

    def forward(self, x):
        x = x[:,:,::self.cut_size]
        return x.contiguous()

class EcaLayer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(EcaLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)#.unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
"""    
self.f_node = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
"""

"""
if not self.fixed_step_solver:
            self.node = NeuralDE(self.f_node, solver=solver_type)
"""

class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=200):
        super().__init__()
        self.d_model = d_model
        # make size eligible for 1d convolutional output: channels before sequence length  
        pe = torch.zeros(d_model, max_seq_len)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[i, pos] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[i + 1, pos] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(-1)
            pe = self.pe[:,:, :seq_len].view(1,self.d_model, seq_len)
            x = x + pe
            return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, seq_len = 1000):
        super(TemporalBlock, self).__init__()
        
        if seq_len is not None:
            self.pos = PositionalEncoder(n_inputs, seq_len)
        else:
            self.pos = None
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, ))
        self.chomp1 = Chomp1d(padding)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout1d(dropout)
        self.node1 = Node(hidden_size=n_outputs)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, ))
        self.chomp2 = Chomp1d(padding)
        
        #self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout1d(dropout)
        self.node2 = Node(hidden_size=n_outputs)

        self.net1 = nn.Sequential(self.conv1, self.chomp1, self.elu, self.dropout1)
        self.net2 = nn.Sequential(self.conv2, self.chomp2, self.elu, self.dropout2) 
        self.attn1 = EcaLayer()
        self.attn2 = EcaLayer()

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.node3 = Node(hidden_size=n_outputs)
        self.gauss = GaussianNoise(sigma=1.0,device="cpu")
        #self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #print(x.size())
        if self.pos is not None:
            x = self.pos(x)
        out_net1 = self.net1(x)
        out_net1 = self.attn1(out_net1)
        #out_net1 = self.gauss(out_net1)
        #out_node1 = self.node1(out_net1, tspan)
        out_net2 = self.net2(out_net1)
        out_net2 = self.attn2(out_net2)
        #out_net2 = self.gauss(out_net2)
        #out_node2 = self.node2(out_net2, tspan)
        #print(torch.isnan(out_node2).sum())
        res = x if self.downsample is None else self.downsample(x)
        out_conv = self.elu(out_net2+res)
        #if self.pos is not None:
        #    out_conv = self.pos(out_conv)
        #out_node = self.node3(out_conv, tspan)
        #print(torch.isnan(out_node2).sum())
        return out_conv

class TemporalBlockWithCutOut(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlockWithCutOut, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=1, dilation=1, ))
        self.chomp1 = Chomp1d(1)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout1d(dropout)
        self.node1 = Node(hidden_size=n_outputs)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=1, dilation=1, ))
        self.chomp2 = Chomp1d(1)
        self.cutout = CutOut1d(dilation)
        
        #self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout1d(dropout)
        self.node2 = Node(hidden_size=n_outputs)

        self.net1 = nn.Sequential(self.conv1, self.chomp1, self.elu, self.dropout1)
        self.net2 = nn.Sequential(self.conv2, self.chomp2, self.elu, self.dropout2) 
        self.attn1 = EcaLayer()
        self.attn2 = EcaLayer()

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.node3 = Node(hidden_size=n_outputs)
        self.gauss = GaussianNoise(sigma=1.0,device="cpu")
        #self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #print(x.size())
        out_net1 = self.net1(x)
        out_net1 = self.attn1(out_net1)
        #out_net1 = self.gauss(out_net1)
        #out_node1 = self.node1(out_net1, tspan)
        out_net2 = self.net2(out_net1)
        out_net2 = self.attn2(out_net2)
        #out_net2 = self.gauss(out_net2)
        #out_node2 = self.node2(out_net2, tspan)
        #print(torch.isnan(out_node2).sum())
        res = x if self.downsample is None else self.downsample(x)
        out_conv = self.elu(out_net2+res)
        out_conv = self.cutout(out_conv)
        #out_node = self.node3(out_conv, tspan)
        #print(torch.isnan(out_node2).sum())
        return out_conv


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, seq_len=1000):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            if i == 0:
                s = seq_len
            else:
                s = None
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, seq_len=s)]

        self.network = sequentialMultiInput(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalCutOutConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalCutOutConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2#2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlockWithCutOut(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = sequentialMultiInput(*layers)

    def forward(self, x):
        return self.network(x)

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
        self.embeddingsLettersInput = nn.Embedding(self.letters_size, self.nb_final_kernels_token, padding_idx=self.letters_pad)
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
        self.token_squeeze = nn.Sequential(
            #nn.Dropout(dropout_value_conv),
            nn.Conv1d(self.nb_kernels_token, self.nb_kernels_token, 
            kernel_size=self.max_len_tok, stride=1, padding=0),
            #Self_Attn_2D(self.nb_kernels_token),
        #    nn.Dropout2d(dropout_value_conv),
        #    nn.MaxPool2d((3, 1), stride=(2, 1), padding=(1,0)),
            ##nn.BatchNorm2d(self.nb_kernels_token),
            EcaLayer(),
            nn.ELU(),
        #    nn.Conv2d(self.nb_kernels_token, self.nb_kernels_token, 
        #              kernel_size=(3,1), stride=(2,1), padding=(2,0)),
            #nn.BatchNorm2d(self.nb_kernels_token),
            #nn.Dropout2d(dropout_value_conv),
        #    nn.ELU(),
        #    nn.Conv2d(self.nb_kernels_token, self.nb_final_kernels_token, 
        #              kernel_size=(kernel_size_02,1), stride=(1,1), padding=(0,0)),
        #    nn.ELU()
            )
        
        #self.token_lin = nn.Linear(kernel_size_02*self.nb_final_kernels_token, nb_final_kernels_token)
        
        #self.minicontext_L = nn.Sequential(
            #nn.Dropout(dropout_value_conv),
        #    nn.Conv2d(1, self.nb_kernels_token, 
        #    kernel_size=(3,self.nb_final_kernels_token), stride=(2,1), padding=(2,0)),
            #Self_Attn_2D(self.nb_kernels_token),
        #    nn.Dropout2d(dropout_value_conv),
        #    nn.MaxPool2d((3, 1), stride=(2, 1), padding=(1,0)),
            ##nn.BatchNorm2d(self.nb_kernels_token),
        #    nn.ELU(),
            #nn.Conv2d(self.nb_kernels_token, self.nb_kernels_token, 
            #          kernel_size=(3,1), stride=(2,1), padding=(2,0)),
            #nn.BatchNorm2d(self.nb_kernels_token),
            #nn.Dropout2d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv2d(self.nb_kernels_token, self.nb_final_kernels_token, 
            #          kernel_size=(kernel_size_01,1), stride=(1,1), padding=(0,0)),
            #nn.ELU()
        #       )
        
        #self.minicontext_L_lin = nn.Linear(kernel_size_02*self.nb_final_kernels_token, nb_final_kernels_token)
        # 
        #self.minicontext_R = nn.Sequential(
            #nn.Dropout(dropout_value_conv),
        #    nn.Conv2d(1, self.nb_kernels_token, 
        #    kernel_size=(3,self.nb_final_kernels_token), stride=(2,1), padding=(2,0)),
            #Self_Attn_2D(self.nb_kernels_token),
        #    nn.Dropout2d(dropout_value_conv),
        #    nn.MaxPool2d((3, 1), stride=(2, 1), padding=(1,0)),
            ##nn.BatchNorm2d(self.nb_kernels_token),
        #    nn.ELU(),
            #nn.Conv2d(self.nb_kernels_token, self.nb_kernels_token, 
            #          kernel_size=(3,1), stride=(2,1), padding=(2,0)),
            #nn.BatchNorm2d(self.nb_kernels_token),
            #nn.Dropout2d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv2d(self.nb_kernels_token, self.nb_final_kernels_token, 
            #          kernel_size=(kernel_size_01,1), stride=(1,1), padding=(0,0)),
            #nn.ELU()
        #    )
        
        #self.minicontext_R_lin = nn.Linear(kernel_size_02*self.nb_final_kernels_token, nb_final_kernels_token)
        
       
        
        kernel_size_1a = (self.max_len_tok+4-(3-1)-1)//2 + 1
        kernel_size_1b = (kernel_size_1a+2-(3-1)-1)//1 + 1
        #kernel_size_2 = (self.context_size+4-(3-1)-1)//2 + 1
        #self.context_size
        kernel_size_2a = (self.context_size+4-(3-1)-1)//2 + 1
        kernel_size_2b = (kernel_size_2a+2-(3-1)-1)//1 + 1
        
        self.context_L = nn.Sequential(
        #    nn.Dropout(dropout_value_conv),
           nn.Conv1d(self.nb_kernels_c_L, self.nb_kernels_c_L, kernel_size=self.max_len_tok, stride=self.max_len_tok, padding=0),
            ##nn.BatchNorm3d(self.nb_kernels_c_L),
            #Self_Attn_3D(self.nb_kernels_c_L),
        #    nn.MaxPool3d((3, 3, 1), stride=(1, 1, 1), padding=(1,1,0)),
        #    nn.Dropout3d(dropout_value_conv),
            EcaLayer(),
            nn.ELU(),
            nn.Conv1d(self.nb_kernels_c_L, self.nb_kernels_c_L, kernel_size=self.context_size, stride=self.context_size, padding=0),
            ##nn.BatchNorm3d(self.nb_kernels_c_L),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_L, self.nb_final_kernels_c_L, kernel_size=(1,kernel_size_1b,1), stride=(1,1,1), padding=(0,0,0)),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_L, self.nb_final_kernels_c_L, kernel_size=(3,1,1), stride=(2,1,1), padding=(2,0,0)),
            #nn.Dropout3d(dropout_value_conv),
            EcaLayer(),
            nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_L, self.nb_final_kernels_c_L, kernel_size=(kernel_size_2,1,1), stride=(1,1,1), padding=(0,0,0)),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            )
        #self.context_L_lin = nn.Linear(kernel_size_1b*kernel_size_2b*self.nb_final_kernels_token, nb_final_kernels_token)
        
        self.context_R = nn.Sequential(
        #    nn.Dropout(dropout_value_conv),
           nn.Conv1d(self.nb_kernels_c_R, self.nb_kernels_c_R, kernel_size=self.max_len_tok, stride=self.max_len_tok, padding=0),
            ##nn.BatchNorm3d(self.nb_kernels_c_L),
            #Self_Attn_3D(self.nb_kernels_c_R),
        #    nn.MaxPool3d((3, 3, 1), stride=(1, 1, 1), padding=(1,1,0)),
        #    nn.Dropout3d(dropout_value_conv),
            EcaLayer(),
            nn.ELU(),
            nn.Conv1d(self.nb_kernels_c_R, self.nb_kernels_c_R, kernel_size=self.context_size, stride=self.context_size, padding=0),
            ##nn.BatchNorm3d(self.nb_kernels_c_L),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_R, self.nb_final_kernels_c_R, kernel_size=(1,kernel_size_1b,1), stride=(1,1,1), padding=(0,0,0)),
            #nn.Dropout3d(dropout_value_conv),
            EcaLayer(),
            nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_R, self.nb_final_kernels_c_R, kernel_size=(3,1,1), stride=(2,1,1), padding=(2,0,0)),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            #nn.Conv3d(self.nb_kernels_c_R, self.nb_final_kernels_c_R, kernel_size=(kernel_size_2,1,1), stride=(1,1,1), padding=(0,0,0)),
            #nn.Dropout3d(dropout_value_conv),
            #nn.ELU(),
            )

        #self.context_R_lin = nn.Linear(kernel_size_1b*kernel_size_2b*self.nb_final_kernels_token, nb_final_kernels_token)
        
        self.input_size_together_1 = self.nb_final_kernels_token+self.nb_final_kernels_minic_L+self.nb_final_kernels_minic_R
        self.input_size_together_2 = self.nb_final_kernels_token + self.nb_final_kernels_c_L + self.nb_final_kernels_c_R
        
        self.together1 = nn.Linear(self.input_size_together_1,
                                   self.nb_final_kernels_token)
        
        self.together2 = nn.Linear(self.nb_final_kernels_token*5,self.nb_final_kernels_token)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.attn_dense = nn.Linear(self.input_size_together_2,self.input_size_together_2, bias=False)

        vector_size_after_convs = (kernel_size_1b*kernel_size_2b*self.nb_final_kernels_token)*2 + kernel_size_02*self.nb_final_kernels_token
        
        
        
        self.encoder_token = TemporalConvNet(
            num_inputs = self.nb_kernels_token,
            num_channels = [self.nb_kernels_token]*5,
            kernel_size = 3,
            dropout = 0.0,
            seq_len=self.max_len_tok
        )
        self.encoder_left_context = TemporalConvNet(
            num_inputs = self.nb_kernels_token,
            num_channels = [self.nb_kernels_token]*8,
            kernel_size = 3,
            dropout = 0.0,
            seq_len=self.context_size*self.max_len_tok
        )
        self.encoder_right_context = TemporalConvNet(
            num_inputs = self.nb_kernels_token,
            num_channels = [self.nb_kernels_token]*8,
            kernel_size = 3,
            dropout = 0.0,
            seq_len=self.context_size*self.max_len_tok
        )
        
        
        self.decoder_lemma = TemporalConvNet(
            num_inputs = self.nb_kernels_token*3,
            num_channels = [self.nb_kernels_token]*5,
            kernel_size = 3,
            dropout = 0.0,
            seq_len = self.max_len_lemma
        )
        self.decoder_pos = TemporalConvNet(
            num_inputs = self.nb_kernels_token*3,
            num_channels = [self.nb_kernels_token]*5,
            kernel_size = 3,
            dropout = 0.0,
            seq_len = self.max_len_lemma
        )
        self.decoder_morph = TemporalConvNet(
            num_inputs = self.nb_kernels_token*3,
            num_channels = [self.nb_kernels_token]*5,
            kernel_size = 3,
            dropout = 0.0,
            seq_len = self.max_len_lemma
        )


        
       
        
        

      
        #for lemma
        if include_lemma:
            #depth_dense_layer_lemma = 2
            #self.layer_lemma =  nn.ModuleList(\
            #[nn.Sequential(nn.Dropout(dropout_value_dense), nn.ELU(), nn.Linear(self.lemma_size, self.lemma_size))\
            # if x != depth_dense_layer_lemma-0\
            # else\
            # nn.Sequential(nn.ELU(), nn.Linear(self.lemma_size, self.lemma_size))
            # for x in range(depth_dense_layer_lemma)])#nn.Linear(self.nb_final_kernels, self.nb_final_kernels)
            self.final_layer_lemma = nn.Linear(self.nb_final_kernels_token, self.lemma_size)
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
            
            #minicontext_L = context_left[-1,:,:].unsqueeze(0)
            #minicontext_R = context_right[0,:,:].unsqueeze(0)
            #print(tokens[1])
            #if not self.embedding:
            #    tokens = self.make_one_hot(tokens, C = self.letters_size)
            #    context_left = self.make_one_hot(context_left, C = self.letters_size)
            #   context_right = self.make_one_hot(context_right, C = self.letters_size)
            #    minicontext_L = self.make_one_hot(minicontext_L, C = self.letters_size)
            #    minicontext_R = self.make_one_hot(minicontext_R, C = self.letters_size)
            #print(context_left.size())
            #quit()
            #else:
            tokens = self.embeddingsLettersInput(tokens).transpose(2,1)#.reshape(-1, self.nb_final_kernels_token, self.max_len_tok)
            context_left = self.embeddingsLettersInputCL(context_left).unsqueeze(1).transpose(4,1).reshape(-1, self.nb_final_kernels_token, self.context_size*self.max_len_tok)
            context_right = self.embeddingsLettersInputCR(context_right).unsqueeze(1).transpose(4,1).reshape(-1, self.nb_final_kernels_token, self.context_size*self.max_len_tok)
            
            #print(tokens)
            #quit()
            #minicontext_L = self.embeddingsLettersInput(minicontext_L).transpose(1,0)#.squeeze()
            #minicontext_R = self.embeddingsLettersInput(minicontext_R).transpose(1,0)#.squeeze()
            
            #print(tokens.size(), context_left.size())
            #quit()
            #tokens = tokens.reshape(self.max_len_tok,-1,self.nb_final_kernels_token).transpose(1,0).unsqueeze(1)
            #context_left = context_left.reshape(-1, 1,self.context_size,self.max_len_tok,self.nb_final_kernels_token) 
            #context_right = context_right.reshape(-1, 1,self.context_size,self.max_len_tok,self.nb_final_kernels_token)
            
            #minicontext_L = minicontext_L.reshape(self.max_len_tok,-1,self.nb_final_kernels_token).transpose(1,0).unsqueeze(1)
            #minicontext_R = minicontext_R.reshape(self.max_len_tok,-1,self.nb_final_kernels_token).transpose(1,0).unsqueeze(1)

            #current_batch_size = tokens.size(0)
            #tokens = self.token(tokens).reshape(current_batch_size, -1)
            
            tokens_encoded = self.encoder_token(tokens)#.max(dim=-1)#.reshape(-1, self.nb_final_kernels_token)
            left_context_encoded = self.encoder_left_context(context_left)#.max(dim=-1)#.reshape(-1, self.nb_final_kernels_token)
            right_context_encoded = self.encoder_right_context(context_right)#.max(dim=-1)#.reshape(-1, self.nb_final_kernels_token)
            
            tokens_encoded = self.token_squeeze(tokens_encoded).reshape(-1, self.nb_final_kernels_token)
            left_context_encoded = self.context_L(left_context_encoded).reshape(-1, self.nb_final_kernels_token)
            right_context_encoded = self.context_R(right_context_encoded).reshape(-1, self.nb_final_kernels_token)
            #print(tokens_encoded.size(), left_context_encoded.size(), 'ffhbfds')
            #tokens = self.elu(self.token_lin(tokens))
            #print(tokens_encoded.size(), left_context_encoded.size())
            
            #quit()
            
            #minicontext_L = self.minicontext_L(minicontext_L).reshape(current_batch_size, -1)
            #minicontext_R = self.minicontext_R(minicontext_R).reshape(current_batch_size, -1)
            #minicontext_L = self.elu(self.minicontext_L_lin(minicontext_L))
            #minicontext_R = self.elu(self.minicontext_R_lin(minicontext_R))
           
            #token_context = torch.cat([minicontext_L, tokens, minicontext_R],1).reshape(-1, self.input_size_together_1)#view(-1,1,1,self.nb_final_kernels*3)#
            
            
            #token_context = self.together1(token_context)
            #token_context = self.elu(token_context)
        
        
            #context_left = self.context_L(context_left).reshape(current_batch_size, -1)#.reshape(-1, self.nb_final_kernels_c_L)
            #context_left = self.elu(self.context_L_lin(context_left))
            
            #print(context_left.size())

            #quit()
            #context_right = self.context_R(context_right).reshape(current_batch_size, -1)#.reshape(-1, self.nb_final_kernels_c_R)
            #context_right = self.elu(self.context_R_lin(context_right))

            mix = torch.cat([tokens_encoded, left_context_encoded, right_context_encoded],1)#.reshape(-1, self.nb_final_kernels_token*3)#view(-1,1,1,self.nb_final_kernels*3) view(-1, self.nb_final_kernels*3)
          
            #attention for dense layer
            #attn_vector = self.softmax(self.tanh(self.attn_dense(mix)))
            #mix = mix * attn_vector
            #mix = self.together2(mix)
            #mix = self.elu(mix)
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

 