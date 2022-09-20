import pandas as pd
import numpy as np

df = pd.read_csv('seq_u.csv')
seq = df['seq']

a = []
[[a.append(x) for x in i if x not in a] for i in seq]
print(a)
print(len(a))

df_b = pd.read_csv('data/seq11.csv')
seq_b = df_b['seq']

b = []
[[b.append(x) for x in i if x not in b] for i in seq_b]
print(b)
print(len(b))

c = []
[c.append(i) for i in a if i not in b]

z = []
[z.append(i) for i in b if i not in a]

print(c)
print(z)

aa = 'A R N D C E Q G H I L K M F P S T W Y V B Z X J'
aa_list = aa.split(' ')
aa_numbers = [x+1 for x in range(25)]
dictionary = {x:y for x,y in zip(aa_list, aa_numbers)}
dictionary[' '] = 0

print(len(seq))
seq = seq.append(seq_b)
print(len(seq))

sequence_list = [[dictionary[i] for i in x] for x in seq]
binarized = [['{0:05b}'.format(i) for i in sequence] for sequence in sequence_list]

print(np.array(binarized))

new_list = []
for i in binarized:
    new_list.append(np.array([[float(char) for char in word] for word in i]))

new = np.array(new_list)
print(new)
print(new.shape)
#sequence_array = np.array(sequence_list)
#print(sequence_array)
np.save('sequence_u.npy', new)