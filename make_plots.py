import json
import matplotlib.pyplot as plt
import os
import sys
import plotly.graph_objects as go
import pandas as pd

cwd = os.getcwd()

# Each element is a list with the jsons for each NN of the question
question = [['VGG1.json', 'VGG2.json', 'VGG3.json', 'VGG16.json'], ['VGG1DA.json', 'VGG2DA.json', 'VGG3DA.json', 'VGG16DA.json'],
['EfficientNetB0.json', 'ResNet50.json', 'EfficientNetB0DA.json', 'ResNet50DA.json'], ['VGG1Panda.json', 'VGG2Panda.json', 'VGG3Panda.json', 'VGG16Panda.json', 'EfficientNetB0Panda.json',
 'ResNet50Panda.json'], ['VGG1PandaUnbalanced.json', 'VGG2PandaUnbalanced.json', 'VGG3PandaUnbalanced.json', 'VGG16PandaUnbalanced.json', 'EfficientNetB0PandaUnbalanced.json',
  'ResNet50PandaUnbalanced.json']]

key_mapping = {
	'train': 'trainHistory',
	'test': 'evalTest'
}

metric_mapping = {
	'loss': 0,
	'accuracy': 1,
	'precision': 2,
	'recall': 3
}

# Function that plots a single metric value for all NN of a question in a single plot
def group_all_train(metric_name,question_number):

	for filename in question[question_number]:
		f = open(filename)
		data = json.load(f'{cwd}\\json\\{f}')

		max_metric = max(max_metric,max(data['trainHistory'][metric_name]))
		plt.plot(range(1,len(data['trainHistory'][metric_name])+1),data['trainHistory'][metric_name],label=f'{filename[:-5]}')
		
	plt.title(f'{metric_name} in {set_name} set')
	plt.xlabel('epoch')
	plt.ylabel(metric_name)
	plt.ylim(0,min(max_metric,1))
	plt.legend(loc="upper right")
	plt.savefig(f'{cwd}\\plots\\group_all_{metric_name}_{question_number}.png',format='png')

def group_all_test(question_number):
	l = []

	for filename in question[question_number]:
		f = open(filename)
		data = json.load(f)

		l.append([f'{filename[:-5]}'] + data['evalTest'][1:])
	
	df = pd.DataFrame(l)
	df.rename(columns = {0: 'Neural Network Name', 1: 'Accuracy', 2: 'Precision', 3: 'Recall'},inplace = True)

	s = pd.plotting.parallel_coordinates(df,'Neural Network Name',color=('#556270', '#4ECDC4', '#C7F464', '#FF1100'))
	fig = s.get_figure()
	fig.savefig('sexy_time.png')
	#fig.savefig(f'{cwd}\\plots\\sexy_time.png',format='png')

	
	'''fig = go.Figure(data = 

		go.Parcoords(
			line = {'color': list(range(len(df['Neural Network Name']))),
			'width': 4,
			'colorscale': 'Electric'},

			dimensions = list([
            {'range': [0,1],
                'label': 'Accuracy', 'values': df['Accuracy']},
            {'range': [0,1],
                'label': 'Precision', 'values': df['Precision']},
            {'range': [0,1],
                'label': 'Recall', 'values': df['Recall']}
        ])
	)
	)

	fig.update_layout(
		plot_bgcolor = 'white',
		paper_bgcolor = 'white'
		)

	fig.show()'''


if __name__ == "__main__":
    set_name = str(sys.argv[1])
    metric_name = str(sys.argv[2])
    question_number = int(sys.argv[3])-1

    if set_name == 'train':
    	group_all_train(set_name,metric_name,question_number)

    else:
    	group_all_test(question_number)

# 1 - VGG1, VGG2, VGG3 & VGG16
# 2 - VGG1DA, VGG2DA, VGG3DA & VGG16DA
# 3 - EfficientNetB0, ResNet50, EfficientNetB0DA, ResNet50DA
# 4 - VGG1Panda, VGG2Panda, VGG3Panda, VGG16Panda, EfficientNetB0Panda, ResNet50Panda
# 5 - VGG1PandaUnbalanced, VGG2PandaUnbalanced, VGG3PandaUnbalanced, VGG16PandaUnbalanced, EfficientNetB0PandaUnbalanced, ResNet50PandaUnbalanced