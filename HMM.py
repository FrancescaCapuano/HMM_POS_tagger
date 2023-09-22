import numpy
from Conll import *
import operator
from nltk.probability import *



class HMM:

	
	'''
	instantiates an HMM object.
	Q is a tuple of states.
	O is a tuple of possible observations.
	aij is a QxQ matrix that stores the transition probabilities.
	a0j is a list that stores the initial probabilities for each state in Q. 
	b is a QxO matrix that stores the emission probabilities.
	'''
	def __init__(self,Q,O,aij,a0i,b):
		self.Q=Q
		self.O=O
		self.aij=aij
		self.a0i=a0i
		self.b=b




	#the attribute bp is initialized to a Qx(O-1) matrix.
	def Backpointers(self,observation):
		self.bp=numpy.zeros((len(self.Q),len(observation)-1))




	'''
	returns a list of the Viterbi values for the last observation. 
	The parameter observation is a list of observations.
	'''
	def Viterbi_p(self,observation):
		V_t=[]
		if len(observation)==1:
			for i in range(len(self.Q)):
				try:
					V=self.a0i[i]*self.b[i][self.O.index(observation[0])]
				except ValueError: #if observation not in Vocab, b is set to 1
					V=self.a0i[i]
				V_t.append(V)
			return V_t
		V_t_minus_1=self.Viterbi_p(observation[:-1])
		for j in range(len(self.Q)):
			Vmax=0
			for i in range(len(self.Q)):	
				try:
					V=V_t_minus_1[i]*self.aij[i][j]*self.b[j][self.O.index(observation[-1])]
				except ValueError: #if observation not in Vocab, b is set to 1
					V=V_t_minus_1[i]*self.aij[i][j]					
				if V>Vmax:
					Vmax=V
					self.bp[j][len(observation)-2]=i
			V_t.append(Vmax)
		self.V_t=V_t
		return self.V_t




	'''
	returns a list with the best sequence of states for the given observation, according to our HMM. 
	The parameter observation is a list of observations.
	'''
	def BestSequence(self,observation):
		self.Backpointers(observation)
		self.Viterbi_p(observation)
		self.best_sequence=[]
		index_j=self.V_t.index(max(self.V_t))
		j=self.Q[index_j]
		self.best_sequence.append(j)
		timesteps=self.bp.shape[1]
		for timestep in reversed(range(timesteps)):
			index_j=self.bp[index_j][timestep]
			self.best_sequence.insert(0,self.Q[int(index_j)])
		return self.best_sequence





#transforms a ProbDist object in a list, given the list of states the rows should correspond to. 
def pdist_to_list(pdist,rows):
	mylist=[]
	for i in range(len(rows)):
		mylist.append(pdist.prob(rows[i]))
	return mylist


'''
transforms a ConditionalProbDist object in a matrix, 
given the list of states the rows should correspond to, 
and the list of symbols the columns should correspond to.
'''
def cpdist_to_matrix(cpdist,rows,colums):	
	matrix=numpy.zeros((len(rows),len(colums)))
	for i in range(len(rows)):
		for j in range(len(colums)):
			matrix[i][j]=cpdist[rows[i]].prob(colums[j])
	return matrix




if __name__=='__main__':


	'''
	if this module is run as the main program,
	we test it on the Eisner's Ice Cream example.
	'''
	O=(1,2,3)
	Q=('H','C')
	a0i=[0.8,0.2]
	aij=numpy.zeros((2,2))
	aij[0][0]=0.7
	aij[0][1]=0.3
	aij[1][0]=0.4
	aij[1][1]=0.6

	b=numpy.zeros((2,3))
	b[0][0]=0.2
	b[0][1]=0.4
	b[0][2]=0.4
	b[1][0]=0.5
	b[1][1]=0.4
	b[1][2]=0.1

	hmm=HMM(Q,O,aij,a0i,b)
	y=[3,1,3]
	print(hmm.BestSequence(y))
