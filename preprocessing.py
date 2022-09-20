import pandas as pd
import numpy as np
'''
df = pd.read_csv('peptides-complete.csv')
df = pd.DataFrame(df)
#print(df['SEQUENCE'][0])

seq_list = df['SEQUENCE']

print(len(seq_list))

result = []
[result.append(x) for x in seq_list if x not in result]

print(len(result))
df = {'seq': result}

df = pd.DataFrame(df)
df.to_csv('seq.csv')'''

aa = 'A R N D C E Q G H I L K M F P S T W Y V B Z X J'
aa_list = aa.split(' ')
print(aa_list)
aa_numbers = [x+1 for x in range(25)]
print(aa_numbers)
dictionary = {x:y for x,y in zip(aa_list, aa_numbers)}
dictionary[' '] = 0
print(dictionary)


df = pd.read_csv('seq.csv')
df = pd.DataFrame(df)
print(len(df['seq']))
sequence = df['seq']
print(sequence[0])
a_list = ['B', 'Z', 'X', 'J']
result_seq = [x for x in sequence if len(x)<=32]
result_seq = [x for x in result_seq if len(x)>=6]
print(len(result_seq))
result = [x[:20] if len(x)>=20 else str(x)+str(' '*int(20-len(x))) for x in result_seq]
print(len(result))
result = [str(x).upper() for x in result]
print(len(result))
for i in a_list:
    result = [str(x) for x in result if i not in str(x)]
print(len(result))


sequence = result
sequence_list = [[dictionary[i] for i in x] for x in sequence]
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
np.save('sequence_6_32.npy', new)
