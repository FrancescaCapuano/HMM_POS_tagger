from HMM import *


class POS_Tagger:



	'''
	instantiates a POS_Tagger object.
	hmm is an HMM object.
	filename is the string of the name of the file to be tagged.
	sentences is a list of lists, each one containing the split words of a sentence.
	'''
	def __init__(self,hmm,filename,sentences):
		self.hmm=hmm
		self.filename=filename
		self.sentences=sentences




	#writes a 'filename_tagged.tt' file to the current directory in the Conll format. 
	def Generate_POS(self):
		with open(self.filename+'_tagged.tt','w') as f:
			for sent in self.sentences:
				sentence_tags=hmm.BestSequence(sent)
				for i in range(len(sent)):
					f.write(sent[i]+'\t'+sentence_tags[i]+'\n')
				f.write('\n')




if __name__=='__main__':


	#asks for the distribution to be used for the emission probs (MLE or Laplace).
	distribution=input('Enter probability distribution:'+'\n')
	while distribution!='MLE' and distribution!='Laplace':
		print('Distribution should be either "MLE" or "Laplace".')
		distribution=input('Enter probability distribution:'+'\n')



	#checks whether train and test files exist, else downloads them.
	train = "de-train.tt"
	test = "de-test.t"

	if Path(test).is_file()==False and Path(train).is_file()==False:
		url='http://www.coli.uni-saarland.de/~koller/materials/anlp/de-utb.zip'
		urllib.request.urlretrieve(url, "de-utb.zip")
		myzip=zipfile.ZipFile('de-utb.zip')
		myzip.extractall()
		myzip.close()




	train=Conll(train)

	#Q corresponds to the tags of the train corpus.
	Q=train.tags

	#O corresponds to the vocabulary of the corpus.
	O=train.vocabulary


	'''
	a0i corresponds to the initial probabilities: a ProbDist object is instantiated 
	from the Conll object with the pdist_initial_tags() method, and then transformed into a list.
	'''
	pdist_initial_tags=train.pdist_initial_tags()
	if pdist_initial_tags.SUM_TO_ONE: #the probs have to sum to 1
		a0i=pdist_to_list(pdist_initial_tags,Q)


	'''
	aij corresponds to the transition probabilities: a ConditionalProbDist object is instantiated 
	from the Conll object with the cpdist_tags_bigrams() method, and then transformed into a matrix.
	'''
	cpdist_tags_bigrams=train.cpdist_tags_bigrams()
	tags=0
	for tag in Q:
		if cpdist_tags_bigrams[tag].SUM_TO_ONE: #the probs have to sum to 1 for each tag
			tags+=1
	if tags==len(Q):
		aij=cpdist_to_matrix(cpdist_tags_bigrams,Q,Q)



	'''
	b corresponds to the emission probabilities: a ProbDist object is instantiated 
	from the Conll object with the cpdist_tags_words() method, and then transformed into a matrix.
	'''
	if distribution=='MLE':
		cpdist_tags_words=train.cpdist_tags_words(distribution)
		tags=0
		for tag in Q:
			if cpdist_tags_words[tag].SUM_TO_ONE: #the probs have to sum to 1 for each tag
				tags+=1
		if tags==len(Q):
			b=cpdist_to_matrix(cpdist_tags_words,Q,O)

	elif distribution=='Laplace':
		cpdist_tags_words=train.cpdist_tags_words(distribution)
		print('Sum of probabilities in b over each tag:') #print to check if the probs sum to 1 for each tag - approximation due to floats
		for tag in Q:
			SUM_TO_ONE=0
			for word in O:
				SUM_TO_ONE+=cpdist_tags_words[tag].prob(word)
			print(tag+': '+str(SUM_TO_ONE))
		b=cpdist_to_matrix(cpdist_tags_words,Q,O)
	

	#instantiates an hmm with the parameters from the training corpus
	hmm=HMM(Q,O,aij,a0i,b)


	#nltk conll corpus reader of the test file.
	test=nltk.corpus.reader.conll.ConllCorpusReader(os.path.dirname(os.path.abspath(__file__)),fileids=test,columntypes=['words'])

	#gets split sentences from test corpus
	sentences=test.sents()

	pos_tagger=POS_Tagger(hmm,'test',sentences)


	print('Writing the POS-tagged file...')

	pos_tagger.Generate_POS()

	print('Done.')