from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import random
import pandas as pd
import numpy as np


records = list(SeqIO.parse("results.fasta", "fasta"))
df = pd.read_csv('seq.csv')
df = pd.DataFrame(df)

result = [str(i.seq) for i in records]
sequence = df['seq']
print(len(result)+len(sequence))
data = []
[data.append(x) for x in sequence if x not in data]
[data.append(x) for x in result if x not in data]

aa = 'A R N D C E Q G H I L K M F P S T W Y V'
aa_list = aa.split(' ')
aa_numbers = [x+1 for x in range(25)]
dictionary = {x:y for x,y in zip(aa_list, aa_numbers)}
dictionary[' '] = 0
print(len(data))
data1 = [x for x in data if 'X' not in str(x).upper()]
print(len(data1))


result_seq = [x for x in data1 if len(x)<=32]
result_seq = [x for x in result_seq if len(x)>=6]
print(len(result_seq))
'''
result = [x[:20] if len(x)>=20 else str(x)+str(' '*int(20-len(x))) for x in result_seq]
print(len(result))
result = [str(x).upper() for x in result]
print(len(result))
'''
no = [x for x in range(len(result_seq))]

d = {'no': no,
     'seq': result_seq}
d = pd.DataFrame(d)
d.to_csv('all.csv')
'''
sequence = result
sequence_list = [[dictionary[i] for i in x] for x in sequence]
binarized = [['{0:05b}'.format(i) for i in sequence] for sequence in sequence_list]

new_list = []
for i in binarized:
    new_list.append(np.array([[float(char) for char in word] for word in i]))

new = np.array(new_list)
print(new.shape)
#sequence_array = np.array(sequence_list)
#print(sequence_array)
np.save('sequence_6_32.npy', new)'''
