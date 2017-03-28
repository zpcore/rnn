import argparse
import numpy as np
import pickle
from scipy import signal as sg

def data_gen():
	parser = argparse.ArgumentParser(description='Generate Training and Testing Samples')
	parser.add_argument('-n', action="store", dest="ts1", type=int)
	parser.add_argument('-t', action="store", dest="ts2", type=int)

	trainingsample = parser.parse_args().ts1
	testingsample = parser.parse_args().ts2

	A=np.array([[0,0,1,0],[0,0,0,1],[-2,1,0,0],[1,-2,0,0]])
	B=np.array([[0,0],[0,0],[1,0],[0,1]])
	C=np.array([[1,0,0,0],[0,1,0,0]])
	D=np.array([[0,0],[0,0]])
	sys=sg.StateSpace(A,B,C,D).to_discrete(0.1)#sample period 0.1s
	print "Two Mass Spring System:\n",sys,'\n'
	x=np.array([[10],[20],[0],[0]],dtype=np.float64)
	ls=[]
	for i in range(trainingsample):
		ls.append(x)	
		f=2*(np.random.rand(2,1)-0.5)#random force to the mass
		x=sys.A.dot(x)+sys.B.dot(f)
	#	y=sys.C.dot(x)+sys.D.dot(f)

	#store the training data in a file for future analysis
	with open(r'./traindata.log', 'wb') as afile:
		pickle.dump(ls, afile)

	#reload object from the file
	"""
	with open(r'./td', 'rb') as _load_file:
		new_d = pickle.load(_load_file)
	"""

	ls=[]
	for i in range(testingsample):
		ls.append(x)
		f=2*(np.random.rand(2,1)-0.5)#random force to the mass
		x=sys.A.dot(x)+sys.B.dot(f)
	#	y=sys.C.dot(x)+sys.D.dot(f)

	#store the training data in a file for future analysis
	with open(r'./testdata.log', 'wb') as afile:
		pickle.dump(ls, afile)


if __name__ == "__main__":
	data_gen()