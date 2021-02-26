import os

def to_gb(b):
    return round(b/float(1<<30),1)

max_gb = 1

with open(f'./data/trainset_{max_gb}gb.txt','w') as f1:       
    with open('./data/trainset.txt','r') as f2:
        for line in f2:
            f1.write(line)
            if to_gb(os.fstat(f1.fileno()).st_size) >= max_gb:
                break

        