import pandas as pd
import sys
import numpy as np

from certify_K import certify_K

v_range = [0, 21]
fn = 'cora'
global_d =  2708 # cora

# fn = 'citeseer'
# global_d = 3327 # citeseer

# fn = 'pubmed'
# global_d = 19717 # pubmed


file = sys.argv[1]
df = pd.read_csv(file, sep="\t")

accurate = df["correct"]
predict = df["predict"]
label = df["label"]
pAHat = df["pABar"]

#print(pAHat)

test_num = predict == label
#print(sum(test_num))
test_acc = sum(test_num) / float(len(label))
#print(sum(df["correct"]))
print('certify acc:', sum(accurate))

alpha = float(sys.argv[2])
print('alpha = {}'.format(alpha))

K = np.zeros(v_range[1], dtype=int)
for idx in range(len(pAHat)):
	if accurate[idx]:
		v = certify_K(pAHat[idx], alpha, global_d, v_range, fn)
		print('pAHat:', pAHat[idx], 'Certified K:', v)
		K[v] += 1
print(K)

K_cer = np.cumsum(K[::-1])[::-1]

for idx in range(len(K_cer)):
	print(idx+1, K_cer[idx])


# fp = open('../thresholds/{}/{}_bi.txt'.format(fn, alpha), 'w')
# for idx in range(len(K_cer)):
# 	print(idx+1, K_cer[idx])
# 	fp.write('{}\t{}\n'.format(idx+1, K_cer[idx]))
# fp.close()

# ##Core: P=0.7  alpha=0.99, N=10K
# (1, 60)
# (2, 60)
# (3, 60)
# (4, 59)
# (5, 59)
# (6, 57)
# (7, 55)
# (8, 55)
# (9, 53)
# (10, 51)
# ...
