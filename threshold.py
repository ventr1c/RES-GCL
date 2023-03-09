# python threshold.py --fn cora --a 70

import numpy as np
import argparse
from decimal import Decimal

parser = argparse.ArgumentParser(description='Get thresholds')
parser.add_argument("--fn", type=str, help="dataset")

parser.add_argument('--r_start', default=0, type=int, 
                    help='[r_start, r_end)')
parser.add_argument('--r_end', default=21, type=int, 
                    help='[r_start, r_end)')
parser.add_argument('--a', type=int, default=80,
                    help='alpha = a / 100')
args = parser.parse_args()

v_range = [args.r_start, args.r_end]


if args.fn == 'cora':
	global_d =  2708 # cora

if args.fn == 'citeseer':
	global_d = 3327 # citeseer

if args.fn == 'pubmed':
	global_d = 19717 # pubmed


a = args.a
b = 100

frac_alpha = float(a) / b
frac_beta = (1 - frac_alpha)
print('alpha = {}, beta = {}'.format(frac_alpha, frac_beta))
print('v =', v_range)

Z = 100
hZ = 50
tZ = 10

alpha = a
beta = b - a
print('alpha = {}, beta = {}'.format(alpha, beta))

fp = open('thresholds/{}/{}_exact.txt'.format(args.fn, frac_alpha), 'w')
fp.write('r\tp_Base\n')

for v in range(v_range[0], v_range[1]):

	half_Z = hZ * (Z ** (global_d - 1))
	total_Z = Z ** global_d

	complete_cnt = []
	cnt = np.load('list_counts/{}/complete_count_{}.npy'.format(args.fn, v), allow_pickle=True)
	complete_cnt += list(cnt)

	raw_cnt = 0
	outcome = []
	for ((m, n), c) in complete_cnt:
		outcome.append((
			# likelihood ratio x flips m, x bar flips n
			# and then count, m, n
			(alpha ** (n - m)) * (beta ** (m - n)), c, m, n
		))
		if m != n:	
			outcome.append((
				(alpha ** (m - n)) * (beta ** (n - m)), c, n, m
			))

		raw_cnt += c
		if m != n:
			raw_cnt += c

	print(v, raw_cnt - 2 ** global_d)
	
	outcome = sorted(outcome, key = lambda x: -x[0])

	p_given = 0
	for i in range(len(outcome)):
		ratio, cnt, m, n = outcome[i]
		p = (alpha ** (global_d - m)) * (beta ** m)
		q = (alpha ** (global_d - n)) * (beta ** n)
		q_delta = q * cnt

		if q_delta < half_Z:
			half_Z -= q_delta
			p_delta = p * cnt
			p_given += p_delta
		else:
			p_delta = p * Decimal(half_Z) / Decimal(q)
			p_given += p_delta
			break

	p_given /= total_Z

	print('v = {}, p_given > {}'.format(v, p_given))
	fp.write('{}\t{}\n'.format(v, p_given))

fp.close()