import matplotlib.pyplot as plt 

#size of training corpus correspond to number of words for 1/3, 2/3 and 3/3 of the number of sentences
plt.plot([80188,172022,264906], [0.8377,0.856,0.8665],'ro')
plt.plot([80188,172022,264906], [0.8377,0.856,0.8665], linewidth=2.0)
plt.ylabel('accuracy')
plt.xlabel('train size (num of words)')
plt.axis([0, 300000, 0.8, 0.9])
plt.show()