import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

matrix = np.load('u_gan/u-gan50.npy')
print(matrix.shape)
print(matrix[0][0])
translate=[]
for z in matrix:
    rounded = [[int(round(float(i), 0)) for i in x] for x in z]
    list_a = [''.join([str(int(x)) for x in i]) for i in rounded]
    a_l = [int(i, 2) for i in list_a]
    translate.append(a_l)

translation = np.array(translate)
print(translation)


def id_generator(i):
    return 'GANcP%s' % (str(i))


a = ' ARNDCEQGHILKMFPSTWYVBZXJ        '

new_translation = [''.join([a[i] for i in x]) for x in translate]
new=[]
id_list=[]
seq_list=[]
numb=0
for i,c in zip(new_translation, range(len(new_translation))):
    z = []
    id = id_generator(str(c))
    id_list.append(id)
    if z not in new:
        numb+=1
    while i.startswith(' ') == True:
        i = i[1:]
    for x in i:
        if x=='B':
            z.append('N')
        elif x=='Z':
            z.append('Q')
        elif x=='X':
            z.append('A')
        elif x=='J':
            z.append('L')
        elif x!=' ':
            z.append(x)
        else:
            continue
    if ''.join(z) not in new:
        new.append(''.join(z))
        sr = SeqRecord(Seq(''.join(z)), id, '', '')
        seq_list.append(sr)
    else:
        continue


print(new_translation)
print(len(new))
print(numb)

with open("u_gan/ugan50.fasta", "w") as output_handle:
    SeqIO.write(seq_list, output_handle, "fasta")