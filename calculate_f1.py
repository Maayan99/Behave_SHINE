import collections
import re
import string
import json
from pathlib import Path

def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	def remove_articles(text):
		regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
		return re.sub(regex, ' ', text)
	def white_space_fix(text):
		return ' '.join(text.split())
	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)
	def lower(text):
		return text.lower()
	return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
	if not s: return []
	return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
	return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
	gold_toks = get_tokens(a_gold)
	pred_toks = get_tokens(a_pred)
	common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
	num_same = sum(common.values())
	if len(gold_toks) == 0 or len(pred_toks) == 0:
		# If either is no-answer, then F1 is 1 if they agree, 0 otherwise
		return int(gold_toks == pred_toks)
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(pred_toks)
	recall = 1.0 * num_same / len(gold_toks)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

def main(args):
	with open(args.input, 'r') as f:
		data_list = json.load(f)
	for data in data_list:
		gold_answers = data['answer']
		if not isinstance(gold_answers, list):
			gold_answers = [gold_answers]
		gold_answers = [str(a) for a in gold_answers]
		output = data['output']
		try:
			# extract the text between '[output_begin] ... [output_end]'
			# pattern = re.compile(r'\[output_begin\](.*)\[output_end\]')
			# search_results = pattern.search(output)
			# if search_results:
			# 	output = search_results.group(1)
			# else:
			# 	pred_answer = output
			pred_answer = output
		except AttributeError:
			pred_answer = ''
		exact_scores = max(compute_exact(a, pred_answer) for a in gold_answers)
		f1_scores = max(compute_f1(a, pred_answer) for a in gold_answers)
		# print(f'gold: {gold_answers}, pred: {pred_answer}, exact: {exact_scores}, f1: {f1_scores}')
		data['prediction'] = pred_answer
		data['scores'] = {
			'exact': exact_scores,
			'f1': f1_scores,
		}
	Path(args.output).parent.mkdir(parents=True, exist_ok=True)
	with open(args.output, 'w') as f:
		json.dump(data_list, f, indent=2)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, required=True)
	parser.add_argument('--output', type=str, required=True)
	args = parser.parse_args()
	main(args)