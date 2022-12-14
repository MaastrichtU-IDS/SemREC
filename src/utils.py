import  requests
from    urllib.request import urlretrieve
import pandas as pd
import gzip
import re

def download_sources(link,folder='/',ext2=None):

    r = requests.get(link)
    for line in r.text.split('\n'):
        for ext in ['.nt','.owl','.txt']:
            if ext in line:
                fname = re.findall('href="(.+'+ext+')"',line)[0]
                print(fname)      
                urlretrieve(url+fname ,'./'+folder+fname)
        
def prepare_subclass_data(data,transitive_classes=None,
                          tc1='class',tc2='subClass',
                          qc1='class_0',qc2='class_1',qc3='class_2',r = 0):
    ''' 
        triplet direction: tc2 -> tc1 
        quadruple direction: qc1 -> qc2 -> qc3
    '''
    node_count = 0
    node_dict = dict({})

    for e in data[tc1]:
        if e not in node_dict.keys():
            node_dict[e] = node_count
            node_count+=1

    for e in data[tc2]:
        if e not in node_dict.keys():
            node_dict[e] = node_count
            node_count+=1      

    # generate triplets and guadruples
    triplets = []
    for triplet in zip(data[tc2],data[tc1]):
        triplets.append((node_dict[triplet[0]],r,node_dict[triplet[1]]))

    quadruples = []
    if transitive_classes is not None:
        for quadruple in zip(transitive_classes[qc1],transitive_classes[qc2],
                             transitive_classes[qc3]):
            quadruples.append((node_dict[quadruple[0]],r,node_dict[quadruple[1]],r,node_dict[quadruple[2]]))
  
    return node_dict, node_count, triplets, quadruples

def prepare_crossclass_data(data_subclass,data_subclass_quads,r1,
                            data_assertion,r2,data_cross_quads,
                          tc1='class',tc2='subClass',
                          ac1='class',ac2='assertion',
                          qc1='class_0',qc2='class_1',qc3='class_2',
                          cc1='assertion',cc2='class_0',cc3='class_1'):
    ''' 
        triplet direction: tc2 -> tc1 
                           ac2 -> ac1
        quadruple direction: qc1 -> qc2 -> qc3
        
        Note: checks that in quad relations, no two entities are the same (self loop)
    '''
                            
    node_count = 0
    node_dict = dict({})
    
    triplets = []
    quadruples = []

    for e in data_subclass[tc1]:
        if e not in node_dict.keys():
            node_dict[e] = node_count
            node_count+=1

    for e in data_subclass[tc2]:
        if e not in node_dict.keys():
            node_dict[e] = node_count
            node_count+=1      

    # generate triplets of type r1
    for triplet in zip(data_subclass[tc2],data_subclass[tc1]):
        triplets.append((node_dict[triplet[0]],r1,node_dict[triplet[1]]))
        
    for e in data_assertion[ac1]:
        if e not in node_dict.keys():
            node_dict[e] = node_count
            node_count+=1

    for e in data_assertion[ac2]:
        if e not in node_dict.keys():
            node_dict[e] = node_count
            node_count+=1      

    # generate triplets of type r2
    for triplet in zip(data_assertion[ac2],data_assertion[ac1]):
        triplets.append((node_dict[triplet[0]],r2,node_dict[triplet[1]]))

    # sub-class relation
    for quadruple in zip(data_subclass_quads[qc1],data_subclass_quads[qc2],
                             data_subclass_quads[qc3]):
            quadruples.append((node_dict[quadruple[0]],r1,node_dict[quadruple[1]],r1,node_dict[quadruple[2]]))
       
    # cross relations
    for quadruple in zip(data_cross_quads[cc1],data_cross_quads[cc2],
                             data_cross_quads[cc3]):
        if quadruple[0] != quadruple[1] and quadruple[1] != quadruple[2]:
            quadruples.append((node_dict[quadruple[0]],r1,node_dict[quadruple[1]],r2,node_dict[quadruple[2]]))
  
    return node_dict, node_count, triplets, quadruples


def load_clg_files(pathfilename):
    
    nt_file = open(pathfilename,'r')
    lines   = nt_file.readlines()
    
    num_lines = 0
    clg_dict  = dict({'s':[],'o':[],'p':[]})
    
    for i,line in enumerate(lines):
        num_lines+=1
        entities = line.split(' ')
        clg_dict['s'].append(entities[0])
        clg_dict['p'].append(entities[1])
        clg_dict['o'].append(entities[2])
        
    df = pd.DataFrame(clg_dict)
        
    #print(num_lines)
    return df
        
    
def load_ore_files(pathfilename):
    
    nt_file = open(pathfilename,'r')
    lines   = nt_file.readlines()
    
    entity_dict = dict({'s':[],'o':[],'p':[]})
    for line in lines:
        parts = line.split(' ')
        entity_dict['s'].append(parts[1])
        entity_dict['p'].append(parts[0])
        entity_dict['o'].append(parts[2][:-1])
        
    df = pd.DataFrame(entity_dict)
    return df
        
def preprocess_data():
    
    # OWL2Bench 
    #sed -i -e 's/(/\ / ; s/)//'
    
    # ORE
    #sed -i -e 's/(/\ / ; s/)//'
    
    # CaLiGraph
    #sed -i -e 's/ \./ \.\n/g' 
    pass


def load_clg_zfiles(pathfilename):
    
    #nt_file = open(pathfilename,'r')
    lines =[]
    with gzip.open(pathfilename,'r') as fin: 
        all_lines = fin.readline()
        lines= all_lines.decode('utf8').split(' .')
    
    num_lines = 0
    clg_dict  = dict({'s':[],'o':[],'p':[]})
    
    for i,line in enumerate(lines):
        num_lines+=1
        entities = line.split(' ')
        if len(entities) != 3: continue
        clg_dict['s'].append(entities[0])
        clg_dict['p'].append(entities[1])
        clg_dict['o'].append(entities[2])
        
    df = pd.DataFrame(clg_dict)
        
    #print(num_lines)
    return df