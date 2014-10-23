import numpy as n
import os
import time

import cPickle
from bisect import bisect

'''
def sampling_from_dist(prob):
	thr = n.random.rand()
	new_topic=0
	tmp = prob[new_topic]
	while tmp < thr:
		new_topic += 1
		tmp += prob[new_topic]
	return new_topic
'''

def sampling_from_dist(prob):
	return bisect( n.cumsum(prob), n.random.rand() )


class at_model:

	def __init__(self, vocab, K, A, docList, authorList, alpha=0.1, eta=0.01):
		"""
		Initialize at_model

		vocab = vocabulary list
		K = number of topics
		A = number of authors
		alpha = author-topic distribution dirichlet parameter
		eta = word-topic distribution dirichlet parameter

		docList
			list of documents, constructed based on the vocab
			format = list(list(words))
			ex) [[0,2,2,3],[1,3,3,4]]
				tokens of 1st document= 0,2,2,3 (note that 2 appears twice becase word 2 used twice in the first document)
		authorList 
			format = list(list(authors))
			at least one author should be exist for each document
			ex) [[0,1],[1,2]] 
				authors of 1st doc = 0, 1
		"""

		self._vocab = vocab
		self._W = len(vocab)
		self._K = K
		self._A = A
		self._D = len(docList)
		self._docList = docList
		self._authorList = authorList
		self._alpha = alpha
		self._eta = eta

		self.c_wt = n.zeros([self._W, self._K])
		self.c_at = n.zeros([self._A, self._K])
		self.topic_assigned = list()
		self.author_assigned = list()
		self.topic_sum = n.zeros(self._K)
		self.author_sum = n.zeros(self._A)

		#initialization
		for di in xrange(0, self._D):
			self.author_assigned.append(list())
			self.topic_assigned.append(list())
			doc = self._docList[di]
			authors = self._authorList[di]
			for wi in xrange(0, len(doc)):
				w = doc[wi]
				#random sampling topic
				#z = n.random.choice(self._K, 1)[0]
				z = int(self._K * n.random.random())
				#random sampling author
				#a = n.random.choice(len(authors),1)[0]
				a = int(len(authors) * n.random.random())

				#assigning sampled value (sufficient statistics)
				self.c_wt[w,z] += 1
				self.c_at[authors[a],z] += 1
				self.topic_sum[z] += 1
				self.author_sum[authors[a]] += 1

				#keep sampled value for future sampling
				self.topic_assigned[di].append(z)
				self.author_assigned[di].append(authors[a])

	def sampling_topics(self, max_iter):
		for iter in xrange(0, max_iter):
			timeBegin = time.time()
			for di in xrange(0, len(self._docList)):
				doc = self._docList[di]
				authors = self._authorList[di]

				for wi in xrange(0, len(doc)):
					w = doc[wi]
					old_z = self.topic_assigned[di][wi]
					old_a = self.author_assigned[di][wi]

					self.c_wt[w, old_z] -= 1
					self.c_at[old_a, old_z] -= 1
					self.topic_sum[old_z] -= 1
					self.author_sum[old_a] -= 1

					wt = (self.c_wt[w, :]+ self._eta)/(self.topic_sum+self._W*self._eta) 
					at = (self.c_at[authors,:] + self._alpha)/(self.author_sum[authors].repeat(self._K).reshape(len(authors),self._K)+self._K*self._alpha)

					#print self.c_at[authors,:]
					#print self.author_sum[authors].repeat(self._K).reshape(len(authors),self._K)
					#print at

					pdf = at*wt
					pdf = pdf.reshape(len(authors)*self._K)
					pdf = pdf/pdf.sum()

					#sampling author and topic
					#need more better way to handle a sample
					#sample = n.random.multinomial(1, pdf)
					#idx = [i for i,x in enumerate(sample) if x==1][0]

					idx = sampling_from_dist(pdf)

					new_ai = idx/self._K
					new_z = idx%self._K

					new_a = authors[new_ai]
					self.c_wt[w,new_z] += 1
					self.c_at[new_a, new_z] += 1
					self.topic_sum[new_z] += 1
					self.author_sum[new_a] += 1
					self.topic_assigned[di][wi] = new_z
					self.author_assigned[di][wi] = new_a
				if di%1000==0:
					print 'iteration : ' + str(iter) + ' doc number: ' + str(di) + '/' + str(len(self._docList))
			timeEnd = time.time()
			print 'Elased time for single iteration : ' + str(timeEnd - timeBegin)
			#for sampletopic in range(0,5):
			#	print [bow[np.argsort(self.c_wt[:,sampletopic])[-1*i]] for i in range(1, 10)]


if __name__ == '__main__':
	#test case
	#atm = at_model([0,1,2,3,4], 2, 3, [[0,0,2,2,3],[1,3,3,4,4]], [[0,1],[1,2]])
	#atm.sampling_topics(10)
	print 'FILE opened, importing pickles'
	bow = cPickle.load(open('bow.pkl', 'rb'))
	d_wi = cPickle.load(open('d_wi.pkl', 'rb'))
	paperid_authorid = cPickle.load(open('paperid_authorid.pkl', 'rb'))
	del d_wi[40314]
	del paperid_authorid[40314]

	print 'Import finished'
	authors = set()
	for v in paperid_authorid.values():
		for v2 in v:
			authors.add(v2)
	print 'Initializing...'
	atm = at_model(bow, len(d_wi), len(authors), d_wi, paperid_authorid.values())
	print 'Inference started'
	atm_results = atm.sampling_topics(100)


	#folder = 'at-result'
	#if not os.path.exists(folder):
	#	os.makedirs(folder)
	#n.savetxt(folder + '/word-topic.dat', atm.c_wt)
	#n.savetxt(folder + '/author-topic.dat', atm.c_at)
