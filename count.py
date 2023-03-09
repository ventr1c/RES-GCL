## python count.py --fn cora --range 21


import numpy as np
from scipy.special import comb, factorial
from time import time
import argparse


parser = argparse.ArgumentParser(description='Compute the number of data points in each region')
parser.add_argument("--fn", type=str, help="dataset")
parser.add_argument("--range", type=int, default=21, help="range of certified perturbation size")
parser.add_argument("--K", type=int, default=1, help="binary data")

args = parser.parse_args()



global_comb = dict()
global_powe = dict()

def my_comb(d, m):
	if (d, m) not in global_comb:
		global_comb[(d, m)] = comb(d, m, exact=True)

	return global_comb[(d, m)]

def my_powe(k, p):
	if (k, p) not in global_powe:
		global_powe[(k, p)] = k ** p

	return global_powe[(k, p)]


def get_count(d, m, n, r, K):
	if r == 0 and m == 0 and n == 0:
		return 1
	# early stopping
	if (r == 0 and m != n) or min(m, n) < 0 or max(m, n) > d or m + n < r:
		return 0

	if r == 0:
		return my_comb(d, m) * my_powe(K, m)
	else:
		c = 0

		# the number which are assigned to the (d-r) dimensions
		for i in range(max(0, n-r), min(m, d-r, int(np.floor((m+n-r) * 0.5))) + 1):
			if (m+n-r) / 2 < i:
				break
			x = m - i
			y = n - i
			j = x + y - r
			# j = 0 ## if K = 1
			# the second one implies n <= m+r
			if j < 0 or x < j:
				continue
			tmp = my_powe(K-1, j) * my_comb(r, x-j) * my_comb(r-x+j, j)
			if tmp != 0:
				tmp *= my_comb(d-r, i) * my_powe(K, i)
				c += tmp

		return c



if __name__ == "__main__":

	if args.fn == 'cora':
		global_d =  2708 # cora

	if args.fn == 'citeseer':
		global_d = 3327 # citeseer

	if args.fn == 'pubmed':
		global_d = 19717 # pubmed


	K = args.K
	r_range = [0, args.range]
	m_range = [0, global_d+1]

	print('fn =', args.fn, 'Range of L0 norm =', r_range, 'm_range =', m_range, 'global_d:', global_d, 'data type:', K)

	real_ttl = (K+1)**global_d

	for r in range(r_range[0],r_range[1]):
		ttl = 0
		complete_cnt = []
		for m in range(m_range[0], m_range[1]):
			start = time()
			for n in range(m, min(m+r, global_d)+1):
				c = get_count(global_d, m, n, r, K)
				if c != 0:
					complete_cnt.append(((m, n), c))
					ttl += c
					# symmetric between d, m, n, r and d, n, m, r
					if n > m:
						ttl += c
			
			if m % 100 == 0:
				print('r = {}, m = {:10d}/{:10d}, ttl ratio = {:.4f}, # of dict = {}'.format(r, m, m_range[1], ttl / real_ttl, len(complete_cnt)))
				print(args.fn, len(global_powe), len(global_comb), time() - start)
		
		np.save('list_counts/{}/complete_count_{}'.format(args.fn, r), complete_cnt)

		del complete_cnt 
		del global_comb, global_powe
		
		global_comb = dict()
		global_powe = dict()