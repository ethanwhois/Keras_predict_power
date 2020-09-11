import json
import os

settings={
	'data': {
		'filename': 'E_total.csv',
		'columns':{
			'E-Total[kWh]'
		},
		'sequence_length': 50,
		'train_test_split': 0.8,
		'normalise': True,
	},
	'training':{
		'epochs': 2,
		'batch_size': 32,
	},
	'model': {
		'loss': 'mse',
		'optimizer':'adam',
		'save_dir':'saved_models',
		'layers': [
			{
				'type':'lstm',
				'neurons': 50,
				'input_timesteps':49,
				'input_dim': 1,
				'return_seq': True,
			},
			{
				'type':'dropout',
				'rate': 0.05,
			},
			{
				'type':'lstm',
				'neurons': 100,
				'return_seq': False,
			},
			{
				'type':'dropout',
				'rate': 0.05,
			},
			{
				'type':'dense',
				'neurons': 1,
				'activation': 'Relu'
			},
		]
	}
}

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

result = json.dumps(settings, default=set_default)

with open(os.path.join(os.path.dirname(__file__),'data','config.json'),'w') as f:
    f.write(result)

