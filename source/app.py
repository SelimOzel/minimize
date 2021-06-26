import datetime
import random
import numpy as np
from scipy.optimize import minimize

from scipyoptimize_hack import dummy_function
from scipyoptimize_hack import _minimize_slsqp

def WhatIsThis(securityList_IN, securityPerc_IN, portfolioValue_IN, date_IN, mode_IN):
	optimizationDate = datetime.datetime(date_IN[0],date_IN[1],date_IN[2])
	securityPriceArray = [0]*len(securityList_IN)
	securityQuantityArray = [1]*len(securityList_IN)

	i = 0
	for securityName in securityList_IN:
		securityPriceArray[i] = random.uniform(0.0, 10.0);
		i = i+1

	# Define objective function
	def func(x_IN, args_IN):
		totalValue = args_IN[0]
		securityPrices = args_IN[1]
		cashPerc = args_IN[2]
		return totalValue*(1-cashPerc) - sum(securityPrices[i]*x_IN[i] for i in range(len(securityPrices)))

	# Define constraints
	cons = []
	for i in range (len(securityList_IN)):
		# Generating a function like this is weird as fuck. But it's related to closure of the environment.
		def funcGenerator(i):
			def constraint(x): return np.array(securityPerc_IN[i] - x[i]*securityPriceArray[i]/portfolioValue_IN)
			return constraint
		cons.append({'type': 'ineq','fun' : funcGenerator(i)} )
	
	##res = minimize(func, securityQuantityArray, args = [portfolioValue_IN, securityPriceArray, securityPerc_IN[len(securityPerc_IN)-1]], constraints=cons, method = 'SLSQP', options={'disp': False})
	res = _minimize_slsqp(func, securityQuantityArray, args = [portfolioValue_IN, securityPriceArray, securityPerc_IN[len(securityPerc_IN)-1]], constraints=cons, method = 'SLSQP', options={'disp': False})

	# Make it integer.
	# Don't forget we are not doing integer programming. 
	# This is just a crude approximation of the integer optimal vector.
	x = np.floor(res.x)

	i = 0
	securityValue = 0
	if(mode_IN == 1):
		print("RESULT")
		for securityName in securityList_IN:
			securityValue += x[i]*securityPriceArray[i]
			print(securityName, "percentage is", (x[i]*securityPriceArray[i]/portfolioValue_IN)*100, "(",securityPerc_IN[i]*100,")", "quantity is", x[i])
			i += 1
		print("Cash percentage is", ((portfolioValue_IN-securityValue)/portfolioValue_IN)*100, "(",securityPerc_IN[len(securityPerc_IN)-1]*100,")")

	return x

def main():
	print("slsqp python entry point")
	WhatIsThis(
		["S1", "S2", "S3", "S4", "S5"], 
		[0.15, 0.15, 0.25, 0.25, 0.20],
		250,
		[2021,6,26], 
		1);

if __name__ == "__main__":
    main()