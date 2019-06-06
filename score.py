import sys
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

predictionsFile = sys.argv[1]
groundtruthFile = sys.argv[2]

# Read predictions
predictions = dict()
with open(predictionsFile,'r') as f:
	reader = csv.reader(f, delimiter=',')
	for line in reader:
		id = line[0]
		score = float(line[1])
		predictions[id] = score

# Read ground truth labels
y_true = []
y_scores  = []
y_pred = []
with open(groundtruthFile,'r') as f:
	reader = csv.reader(f, delimiter=',')

	for line in reader:
		id = line[0]
		label = float(line[1])
		y_true.append(label)

		# Get prediction for this id
		if id in predictions:
			y_scores.append(predictions[id])
			if predictions[id] < 0.5:
				y_pred.append(0)
			else:
				y_pred.append(1)
		else:
			# If there is no prediction for this id, predict 0 by default
			print('No prediction found for sequence with id %s. Default prediction is 0.'%id)
			y_scores.append(0)
			y_pred.append(0)

accuracy = accuracy_score(y_true, y_pred)
aucroc = roc_auc_score(y_true, y_scores)
print('Accuracy: %f'%accuracy)
print('AUC ROC: %f'%aucroc)
