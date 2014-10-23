cd Downloads/coauthor_datasets/

#from os import listdir
#from os.path import isfile, join
#onlyfiles = [ f for f in listdir('./') if isfile(join('./',f)) ]

import re
import numpy as np

docs = {}

fromwhen = 2008

f = open('outputacm.txt', 'r')
t = f.readlines()
for v in t:
	if v[:2] == '#*':
		doc = v.split()
		
	elif v[:2] == '#@':
		author = v
	elif v[:2] == '#t':
		t = int(re.findall(r'[0-9]+', v)[0])
	elif v[:2] == '#!':
		if len(v.split())>1:
			if t> fromwhen:
				doc += v.split()
				docs[author] = doc
	else :
		continue

voca = Vocabulary(docs.values())
bow = voca.return_bagofwords()
id2token = {k:v for k,v in enumerate(bow)}
token2id = {v:k for k,v in enumerate(bow)}

d_wi = voca.document_wordindex(bow)

def return_gensim_corpus(d_wi):
	corpus = []
	for v in d_wi:
		c = [(v2, v.count(v2)) for v2 in v]
		corpus.append(c)
	return corpus

corpus = return_gensim_corpus(d_wi)

import logging, gensim, bz2
logging.basicConfig(filename = '/Users/jooyeon/Desktop/git/dblp/outputacm_from2008.log', filemode = 'a', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2token, num_topics=100, update_every=0, passes=20)


paperid_authors = {}

for k,v in enumerate(docs.keys()):
	authors = list(set(v.split(',')))
	pattern = re.compile('[^a-z-_]+')
	authors = [pattern.sub('', v2.lower()) for v2 in authors]
	try:
		authors.remove('')
	except:
		haha = 1
	paperid_authors[k] = authors

from collections import defaultdict
authors_paperid=defaultdict(list)
for k,v in paperid_authors.items():
	for v2 in v:
		authors_paperid[v2].append(k)

authors = authors_paperid.keys()
authors_id = {v:k for k,v in enumerate(authors)}

paperid_authorid = {k: [authors_id[v2] for v2 in v] for k,v in paperid_authors.items()}
authorid_paperid = {authors_id[k]:v for k,v in authors_paperid.items()}

num_authors = len(authors_paperid.keys())
num_papers = len(paperid_authors.keys())

'''
topic9_paper = [0]*num_papers
for i in range(num_papers):
	topic9_val = [v for v in lda[corpus[i]] if v[0]==9]
	if len(topic9_val) != 0:
		topic9_paper[i] = topic9_val[0][1]
'''

num_topics = 100
topicn_paper = np.zeros((num_topics, num_papers))
for i in xrange(num_papers):
	topicn_paper[:, i] = np.array([v[1] for v in lda[corpus[i]]])
	if i%1000==0:
		print i



'''
w_paper_author = np.zeros((num_papers, num_authors))
for k,v in paperid_authorid.items():
	for v2 in v:
		w_paper_author[k][v2] = np.random.random()
'''

w = []
for i in xrange(num_topics):
	w_paper_author = {k : {v2: 1/float(len(v)) for v2 in v} for k,v in paperid_authorid.items()}
	for k,v in w_paper_author.items():
		s = topicn_paper[i][k]
		#s2 = sum([val for key, val in w_paper_author[k].items()])
		w_paper_author[k] = {k2: v2*s for k2, v2 in v.items()}
	w.append(w_paper_author)
	print i

topic_author = []

#haha = []
#hihi = []

for topic in range(100):
#topic = 7
	for iter in range(50):
		topicn_author = np.zeros(num_authors)
		#haha.append(w[topic][634][79820])

		for i in paperid_authorid.keys():
			#if topic9_paper[i] == 0:
			#	continue
			w_sum = sum([v for v in w[topic][i].values()])
			#print sum([v for v in w_paper_author[19].values()])
			for v in paperid_authorid[i]:
				topicn_author[v] += w[topic][i][v]
				#print v, topic0_author[v]


		for k,v in paperid_authorid.items():

			author_sum = sum([topicn_author[v2] for v2 in v])
			tp = topicn_paper[topic][k]
			for v2 in v:
				w[topic][k][v2] = topicn_author[v2] * tp / author_sum
				#print author_sum
		#hihi.append(topicn_author[79820])
		#print topicn_author[79820]
		#print w[topic][634][79820]
		#print
	topic_author.append(topicn_author)
	print 'Topic : ' + str(topic)
		
		





