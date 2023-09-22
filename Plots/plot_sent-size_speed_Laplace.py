import matplotlib.pyplot as plt 

plt.plot([6,14,22,30,38,46,54,62], [6/25.4,14/30.7,22/40.0,30/44.3,38/46.1,46/46.3,54/61.5,62/65.8],'ro')
plt.plot([6,14,22,30,38,46,54,62], [6/25.4,14/30.7,22/40.0,30/44.3,38/46.1,46/46.3,54/61.5,62/65.8],linewidth=2.0)
plt.ylabel('speed (words/second)')
plt.xlabel('sentence length (num of words)')
plt.axis([0, 65, 0,1.2])
plt.show() 