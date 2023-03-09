import math
scale = 0.1
alpha = 0.05
beta = 0.01
p_hat = 0.8

r = scale * math.sqrt(2*math.log(1/(alpha*beta)))/(p_hat - beta)
# r = 0.1 * math.sqrt(2 * math.log(1 / (0.05 * 0.01))) / (0.8 - 0.01) 

print(r)