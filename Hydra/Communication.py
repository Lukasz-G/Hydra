            
import numpy as np

import torch, sys, math, time

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


#global use_cuda

#use_cuda = torch.cuda.is_available()


######################################################
#
#
#
'''auxilary functions for optional time measurement of communication processes'''
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
#
#
def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))
######################################################


'''a communication module for exchange and integration of a model's parameters between processes while training'''

class Info_exchange():
    def __init__(self,
                 
            model=None,
            alpha=None,
            rank=0,
            size=1,
            active_workers=None,
            use_cuda=False
            
                 ):
        
            self.model=model
            self.alpha=alpha
            self.rank=rank
            self.size=size
            self.active_workers=active_workers
            self.use_cuda=use_cuda
            
            self.msgs=[]#a list of received but not-yet-processes messages
            self.update_done=False#an auxiliary variable to reset learning rate decay
  
  
    def receiver(self):
        ''' a simple function for checking and receiving incoming messages '''
        if comm.Iprobe(source=MPI.ANY_SOURCE, tag=0):
           
            data = comm.recv(buf = None, source=MPI.ANY_SOURCE, tag=0)
            
            self.msgs.append(data)
            
        
       
   
    
    def update(self):
        '''an update function for (a list of) data to update the model according to the GoSGD-protocol'''
        
        if self.msgs:
            
            
            for msg in self.msgs:
                if msg:
                    weights, alpha_2 = msg
                    
                    mix_1 = self.alpha / (alpha_2 + self.alpha)
                    mix_2 = alpha_2 / (alpha_2 + self.alpha)
                    
                    
                    
                    for x_id, param in enumerate(self.model.parameters()):
                        
                        if self.use_cuda:
                            weight = weights[x_id].cuda()
                            param.data = torch.mul(param.data, mix_1) + torch.mul(weight, mix_2)
                            
                            
                        else:
                            param.data = torch.mul(param.data,mix_1) + torch.mul(weights[x_id],mix_2)
                    
                    
                    self.alpha += alpha_2
                    
            self.update_done = True
            
        self.msgs = []
        return
       
        
        

    def sender(self):
        
        '''a sender function conforming the GOSGD-protocol combined together with the receiving module (to prevent a deadlock)'''
        
        dest_rank = self.rank
        
        while dest_rank==self.rank:
            dest_rank_position = np.random.randint(low=0,high=len(self.active_workers))
            dest_rank = self.active_workers[dest_rank_position]
        msg_weights = []
        
        for param in self.model.parameters():
            msg_weights.append(param.data.cpu())

        self.alpha = self.alpha/2 
        
        msg = (msg_weights, self.alpha)
        
    
        
        req_info = comm.issend(msg, dest = dest_rank, tag=0)
   
        
        while not MPI.Request.Test(req_info):
            status = MPI.Status()
            if comm.Iprobe(source=MPI.ANY_SOURCE, tag=0, status = status):
              
                data = comm.recv(buf = None, source=MPI.ANY_SOURCE, tag=0, status = status)
                self.msgs.append(data)
            
        
        
        return
                
       