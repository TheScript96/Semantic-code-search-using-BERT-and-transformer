from flask import Flask, render_template, request, url_for

app = Flask(__name__)

import tensorflow as tf

import pandas as pd

import nmslib

import re

from transformers import AlbertTokenizer, TFAlbertModel

albert_tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2") 

from transformers import  AlbertConfig

config = AlbertConfig.from_pretrained('./albert', output_hidden_states=True)

model = TFAlbertModel.from_pretrained('./albert', config=config,  from_pt=True)

df = pd.read_csv('final_search.csv')

search_index = nmslib.init(method='hnsw', space='cosinesimil')


search_index.loadIndex('./final.nmslib')


def search(query):
	e = albert_tokenizer.encode(query.lower())
	input = tf.constant(e)[None, :] 
	output = model(input)
	v = [0]*768
	for i in range(-1, -13, -1):
		v = v + output[2][i][0][0] 
	emb = v/12
	idxs, dists = search_index.knnQuery(emb, k=5)
	all_funcs = []
	list_of_dist = []
	list_of_git = []
	for idx, dist in zip(idxs, dists):
		if(float(dist)>0.05):
			continue
		code = df['original_function'][idx]
		list_of_dist.append(dist)
		list_of_git.append(df['url'][idx])
		code = re.sub(r'"""(.*)?"""\s\n',r' ',code,flags=re.DOTALL)
		all_funcs.append(code)
	return all_funcs,list_of_dist,list_of_git


@app.route('/')
def main_page():
    return render_template("main_page.html")

@app.route('/results', methods=['GET'])
def results_page():
	query = request.args.get('query')
	funcs,dists,gits = search(query)
	values = len(funcs)
	return render_template("results_page.html",data=query, result = values, codes=funcs,dist=dists,git=gits)

if __name__ == "__main__":
    app.run()
