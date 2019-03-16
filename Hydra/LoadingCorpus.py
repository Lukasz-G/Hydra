
import codecs, os, random, time
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from collections import Counter
#from Hydra.Utils import leave_only_alphanumeric, mark_tags_for_multitoken


''' a function for pruning a text from non-alphanumeric symbols '''

def leave_only_alphanumeric(string):
    return ''.join(ch if ch.isalnum() else ' ' for ch in string)



def mark_tags_for_multitoken(tag_list):#lemma,tag,morph
    length = len(tag_list)
    new_list = []
    for nb, tag in enumerate(tag_list):
        if nb == 0:
            new_list.append(tag + '>')
        if nb == length-1:
            new_list.append('<' + tag)
        elif nb != 0 and nb != length-1:
            new_list.append('<' + tag + '>')

    return new_list

'''Module for loading and formating the training and validation corpus'''

class Loading_and_formating():
    
    def __init__(self,
                directory='corpus',
                nb_instances=None,
                include_lemma=True,
                include_morph=False,
                include_pos=True,
                v2u=True,
                min_lem_cnt=1,
                min_tok_cnt=1,
                num_splits=None,
                name_of_frag=None
                 
                 ):
        self.directory = directory
        self.nb_instances = nb_instances
        self.include_lemma = include_lemma 
        self.include_morph = include_morph        
        self.include_pos = include_pos
        self.v2u = v2u
        self.min_lem_cnt = min_lem_cnt
        self.min_tok_cnt = min_tok_cnt
        self.name_of_frag = name_of_frag
        self.num_splits = num_splits
        self.nb_multitokens= 0
        self.merging_coefficient = 0.0
        
        
        self.instances = self.load_corpus_folder()
        self.length_of_corpus = len(self.instances['token'])
        
    
    def load_file(self, filepath = None):
        
        instances = {'token': []}
        if self.include_lemma:
            instances['lemma'] = []
        if self.include_pos:
            instances['pos'] = []
        if self.include_morph:
            instances['morph'] = []
        open_file = codecs.open(filepath, 'r', encoding='utf8')
        for line in open_file:
            # print(line)
            line = line.strip()
            if line and not line[0] == '@':
                try:
                    comps = line.split()
                    tok = comps[0].lower().strip()
                    
                    if self.include_lemma:
                        lem = comps[1].lower().strip().replace('-', '')
                    if self.include_pos:
                        pos = comps[2]
                    if self.include_morph:
                        # morph = '|'.join(sorted(set(comps[3].split('|'))))
                        morph = '|'.join(comps[3].split('|'))
                    
                    
                    if '~' in tok:
                        self.nb_multitokens += 1
                        toks = tok.split('~')
                        nb_tok = len(toks)
                        toks_=[]
                        for t in toks:
                            t = leave_only_alphanumeric(t)
                            t = t.strip().replace('~', '').replace(' ', '')
                            if t == '': #for &-like cases
                                t = ' '
                            if self.v2u:
                                t = t.replace('v', 'u')
                            toks_.append(t)
                        instances['token'].extend(toks_)
                        if self.include_lemma:
                            lem = mark_tags_for_multitoken([lem]*nb_tok)
                        if self.include_pos:
                            pos = mark_tags_for_multitoken([pos]*nb_tok)
                        if self.include_morph:
                            # morph = '|'.join(sorted(set(comps[3].split('|'))))
                            morph = mark_tags_for_multitoken([morph]*nb_tok)
                        if self.include_lemma:
                            instances['lemma'].extend(lem)
                        if self.include_pos:
                            instances['pos'].extend(pos)
                        if self.include_morph:
                            instances['morph'].extend(morph)
                        
                        
                        
                    else:
                        tok = leave_only_alphanumeric(tok)
                        if self.v2u:
                            tok = tok.replace('v', 'u')
                        tok = tok.replace(' ', '')
                        if tok == '': #for &-like cases
                            tok = ' '
                        instances['token'].append(tok)
                    
                    
                    
                    
                        if self.include_lemma:
                            instances['lemma'].append(lem)
                        if self.include_pos:
                            instances['pos'].append(pos)
                        if self.include_morph:
                            instances['morph'].append(morph)
                except Exception as e:
                    print(filepath, ':', line, ':',e)
            if self.nb_instances:
                if len(instances['token']) >= self.nb_instances:
                    break
        
        open_file.close()
        
        return instances
        
    def load_corpus_folder(self):
        
        self.instances = {'token': []}
        if self.include_lemma:
            self.instances['lemma'] = []
        if self.include_pos:
            self.instances['pos'] = []
        if self.include_morph:
            self.instances['morph'] = []
        for root, dirs, files in os.walk(self.directory):
            for name in files:
                filepath = os.path.join(root, name)
    
                if not filepath.endswith('txt'):
                    continue
                #print(filepath)
                insts = self.load_file(filepath=filepath)
                self.instances['token'].extend(insts['token'])
                if self.include_lemma:
                    self.instances['lemma'].extend(insts['lemma'])
                if self.include_pos:
                    self.instances['pos'].extend(insts['pos'])
                if self.include_morph:
                    self.instances['morph'].extend(insts['morph'])
        
        return self.instances
    
    def index_characters1(self):
        
        self.tokens = self.instances['token']
      
        vocab = {ch for tok in self.tokens for ch in tok.lower()}
       
        char_vocab = tuple(sorted(vocab))
        self.char_vector_dict, self.char_idx = {}, {}
        filler = np.zeros(len(char_vocab), dtype='float32')
        
        self.number_char = 0
        for idx, char in enumerate(char_vocab):
            ph = filler.copy()
            ph[idx] = 1
            self.char_vector_dict[char] = ph
            self.char_idx[idx] = char
            self.number_char += 1
    
        return self.char_vector_dict, self.char_idx, self.number_char
    
    def index_characters2(self):
        
        if not self.char_idx:
            self.tokens = self.instances['token']
            vocab = {ch for tok in self.tokens for ch in tok.lower()} 
            char_vocab = tuple(sorted(vocab))
            self.char_idx = {}
            for idx, char in enumerate(char_vocab):
                self.char_idx[idx] = char
        characters = []
        for idx, let in self.char_idx.iteritems():
            characters.append(let)
        label_encoder = LabelEncoder()
        #one_hot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = label_encoder.fit_transform(characters + ['<UNK>'] +['<PAD>'])
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        #one_hot_encoded = one_hot_encoder.fit_transform(len(integer_encoded),1)
        self.char_encoder = label_encoder#, one_hot_encoder

        return self.char_encoder
    
    
    def index_lemmas(self):
        
        self.lemmas = self.instances['lemma']
        lemmas_singled = []
        lemmas_unique = []
        for lemma in self.lemmas:
            lemmas_unique.append(lemma)
            single_lemma= lemma.split('+')
            lemmas_singled.extend(single_lemma)
        
        self.cnt = Counter(lemmas_singled)
        #trunc_lems = [k for k, v in cnt.items() if v >= self.min_lem_cnt]
        trunc_lems = []
        
        for k, v in self.cnt.items():
            if v >= self.min_lem_cnt: 
                trunc_lems.extend(list(k))
        
        self.lemmas_unique = set(lemmas_unique)
        self.lemma_encoder = LabelEncoder()
        self.lemma_encoder.fit(trunc_lems + ['<UNK>'] + ['<PAD>']+ ['+']+['<SOS>'])
        
        #print(self.lemma_encoder.classes_)
        #quit()
        
        return self.lemma_encoder
    
    
    
    def index_pos(self):
        self.pos = self.instances['pos']
        pos_singled = []
        for pos_ in self.pos:
            single_pos= pos_.replace('+',' ').replace('>',' ').replace('<',' ').split()
            pos_singled.extend(single_pos)
       
        
        self.pos_encoder = LabelEncoder()
        self.pos_encoder.fit(pos_singled + ['<UNK>'] + ['<PAD>']+ ['<']+['>']+['+'])
        return self.pos_encoder
    
    def index_morph(self):
        self.morph = self.instances['morph']
        morph_singled = []
        for morph_ in self.morph:
            single_morph= morph_.replace('+',' ').replace('>',' ').replace('<',' ').split()
            morph_singled.extend(single_morph)
            
        self.morph_encoder = LabelEncoder()
        self.morph_encoder.fit(morph_singled + ['<UNK>'] + ['<PAD>']+ ['<']+['>']+['+'])
        return self.morph_encoder
    
    def index_tokens(self):
        
        if not self.instances:
            self.instances = Loading_and_formating().load_annotated_file()
        
        tokens = self.instances['token']
        
        cnt = Counter(tokens)
        self.trunc_lems = [k for k, v in cnt.items() if v >= self.min_tok_cnt]
        self.tok_encoder = LabelEncoder()
        self.tok_encoder.fit(tokens + ['<UNK>'])
        return self.tok_encoder
    
    def count_multilabel_lemma(self):
        
        lemmas = self.instances['lemma']
        multilabes = []
        for lemma in lemmas:
            if '+' in lemma:
                multilabes.append(lemma)
        
        self. multilabes = list(set(multilabes))
        
        return self. multilabes
    
    
    
    
    def change_utf8(self, string):
        try:
            string = string.encode('utf-8')
            return string
        except:
            return string
    
    def create_fragments(self):
        
        fragment = {}
        print(len(self.instances['token']), self.num_splits)
        k, m = divmod(len(self.instances['token']), self.num_splits)
        for x in xrange(self.num_splits):
            #file_for_frag = os.path.join(self.frags_folder_name,self.name_of_frag)
            file_to_save = open('{}_{}.txt'.format(self.name_of_frag, x), 'wb')
            s, e = x * k + min(x, m), (x + 1) * k + min(x + 1, m)
            lemmas, poss, morphs = None, None, None
            tokens = self.instances['token'][s:e]
            
            if self.include_lemma:    
                lemmas = self.instances['lemma'][s:e]                      
            else:
                lemmas = [None] 
            if self.include_pos:
                poss = self.instances['pos'][s:e]
            else:
                poss = [None]
            if self.include_morph:
                morphs = self.instances['morph'][s:e]   
            else:
                morphs = [None]  
            
            fragment = map(None, tokens, lemmas, poss, morphs)
            
            for f in fragment:
                
                tokens, lemmas, poss, morphs = [self.change_utf8(instance) for instance in f]
                #tokens, lemmas, poss, morphs = f
                
                
                file_to_save.write(('{}\t{}\t{}\t{}\t'.format(tokens, lemmas, poss, morphs)))
                file_to_save.write('\n')
            
            file_to_save.close()
            print('created file no: {}'.format(x))
            
    def create_dev_file(self):
        #file_for_frag = os.path.join(self.frags_folder_name,self.name_of_frag)
        file_to_save = open('{}.txt'.format(self.name_of_frag), 'wb')
        tokens = self.instances['token'] 
        if self.include_lemma:    
            lemmas = self.instances['lemma']
        else:
            lemmas = [None]
        if self.include_pos:
            poss = self.instances['pos']
        else:
            poss = [None]
        if self.include_morph:
            morphs = self.instances['morph']
        else:
            morphs = [None]  
        
        fragment = map(None, tokens, lemmas, poss, morphs)
        
        for f in fragment:  
            #tokens, lemmas, poss, morphs = f
            tokens, lemmas, poss, morphs = [self.change_utf8(instance) for instance in f]
            file_to_save.write(('{}\t{}\t{}\t{}'.format(tokens, lemmas, poss, morphs)))
            file_to_save.write('\n')
        file_to_save.close()
       
        return
    
    
    
    #TODO merging tokens /splitting tokens
    
    def check_sanity_index_merge_split(self, random_index):
        #if 
        if self.instances['lemma']:
            try:
                instance_ = self.instances['lemma'][random_index]
            except:
                return False
            if ('+' in instance_) or ('<' in instance_) or ('>' in instance_):
                return False
            else:
                return True
        elif self.instances['pos']:
            try:
                instance_ = self.instances['pos'][random_index]
            except:
                return False
            if ('+' in instance_) or ('<' in instance_) or ('>' in instance_):
                return False
            else:
                return True
        elif self.instances['morph']:
            try:
                instance_ = self.instances['morph'][random_index]
            except:
                return False
            if ('+' in instance_) or ('<' in instance_) or ('>' in instance_):
                return False
            else:
                return True
        
    
    def selecting_other_index(self, already_selected, length_corpus):
        
        #randomly
        '''
        random.seed(time.time())
        random_index = random.randrange(length_corpus)
        if random_index != already_selected:
            return random_index
        else:
            return self.selecting_other_index(already_selected, length_corpus)
        '''
        #neighbourhood
        
        random.seed(time.time())
        random_index = random.randrange(2)
        if random_index == 0:
            return random_index-1
        else:
            return random_index+1
        
        
        
    def lemma_tags_merger(self, random_index, another_index):
        
        if random_index < another_index:
            self.instances['token'][random_index] = self.instances['token'][random_index] + self.instances['token'][another_index]
            del self.instances['token'][another_index]
            if self.instances['lemma']:
                self.instances['lemma'][random_index] = self.instances['lemma'][random_index] + '+' + self.instances['lemma'][another_index]
                del self.instances['lemma'][another_index]
            if self.instances['pos']:
                self.instances['pos'][random_index] = self.instances['pos'][random_index] + '+' + self.instances['pos'][another_index]
                del self.instances['pos'][another_index]
            if self.instances['morph']:
                self.instances['morph'][random_index] = self.instances['morph'][random_index] + '+' + self.instances['morph'][another_index]
                del self.instances['morph'][another_index]
        
        if random_index > another_index:
            self.instances['token'][random_index] = self.instances['token'][another_index] + '+' + self.instances['token'][random_index]
            del self.instances['token'][another_index]
            if self.instances['lemma']:
                self.instances['lemma'][random_index] = self.instances['lemma'][another_index] + '+' + self.instances['lemma'][random_index]
                del self.instances['lemma'][another_index]
            if self.instances['pos']:
                self.instances['pos'][random_index] = self.instances['pos'][another_index] + '+' + self.instances['pos'][random_index]
                del self.instances['pos'][another_index]
            if self.instances['morph']:
                self.instances['morph'][random_index] = self.instances['morph'][another_index] + '+' + self.instances['morph'][random_index]
                del self.instances['morph'][another_index]
    
    def lemma_tags_splitter(self, random_index):
        token_in_question = self.instances['token'][random_index]
        random.seed(time.time())
        splitting_index = 1
        while splitting_index < 2 or splitting_index+1 > len(token_in_question)-2:
            splitting_index = random.randrange(len(token_in_question))
        part_one = token_in_question[splitting_index:]
        part_two = token_in_question[:splitting_index]
        self.instances['token'][random_index] = part_one        
        self.instances['token'].instert(random_index+1, part_two)
        if self.instances['lemma']:
            self.instances['lemma'].instert(random_index+1, self.instances['lemma'][random_index])
        if self.instances['pos']:
            self.instances['pos'].instert(random_index+1, self.instances['pos'][random_index])
        if self.instances['morph']:
            self.instances['morph'].instert(random_index+1, self.instances['morph'][random_index])
    

    
    def merging_tokens(self):
        
        if self.merging_coefficient:
            going_through_list = True
        nb = 0.0
        length_corpus = len(self.instances['token'])
        indices_changed = []
        max_hits = self.merging_coefficient*len(self.instances['token'])
        while going_through_list:
            if max_hits == nb:
                break
            random.seed(time.time())
            random_index = random.randrange(length_corpus)
            #instance_ = self.instances['lemma'][random_index]
            answer_sanity = self.check_sanity_index_merge_split(random_index)
            if answer_sanity:
                another_index = self.selecting_other_index(random_index, length_corpus)
                answer_sanity = self.check_sanity_index_merge_split(another_index)
                if not answer_sanity:
                    continue
                self.lemma_tags_merger(random_index, another_index)
                indices_changed.append(random_index)
            else:
                continue
        
    def splitting_tokens(self):
        
        if self.merging_coefficient:
            going_through_list = True
        nb = 0.0
        length_corpus = len(self.instances['token'])
        indices_changed = []
        max_hits = self.merging_coefficient*len(self.instances['token'])
        while going_through_list:
            if max_hits == nb:
                break
            random.seed(time.time())
            random_index = random.randrange(length_corpus)
            #instance_ = self.instances['lemma'][random_index]
            if len(self.instances['token'][random_index]) <= 3:
                continue
            answer_sanity = self.check_sanity_index_merge_split(random_index)
            if answer_sanity:
                if not answer_sanity:
                    continue
                self.lemma_tags_splitter(random_index)
                indices_changed.append(random_index)
            else:
                continue
        
        
        
        
        
        
        
        
        
           