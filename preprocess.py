import os
import sys
import pandas as pd
import numpy as np
import pickle
import json

from matplotlib import pyplot as plt

from src.folderconstants import *
from shutil import copyfile

#datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB']
datasets = ['UCR']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	if dataset == 'GEC':
		dataset_folder = 'data/GEC'
		df1 = pd.read_csv(os.path.join(dataset_folder, '1_gecco2018_water_quality.csv'))
		np_list = [0,1,2,3,5]
		data = df1.iloc[:, 3:-1].values
		data_list = []
		for i in range(len(np_list)):
			data_list.append(data[:, np_list[i]])
		data = np.stack(data_list, axis=1)
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				if np.isnan(data[i, j]):
					data[i, j] = data[i - 1, j] if i > 0 else 0
		k = 0.08
		for i in range(data.shape[1]):
			if i == 1:
				k = 0.06
			else:
				k = 0.08
			min_temp, max_temp = np.min(data[:, i]), np.max(data[:, i])
			data[:, i] = (data[:, i] - min_temp) / (max_temp - min_temp)
			mean_temp = np.mean(data[:, i])
			lower_bound = mean_temp - k
			upper_bound = mean_temp + k
			data[:, i] = np.clip(data[:, i], lower_bound, upper_bound)
			min_temp, max_temp = np.min(data[:, i]), np.max(data[:, i])
			data[:, i] = (data[:, i] - min_temp) / (max_temp - min_temp)

		train = data[:11000, :]
		test = data[35000:91000, :]

		for i in range(test.shape[1]):
			min_temp, max_temp = np.min(test[:, i]), np.max(test[:, i])
			test[:, i] = (test[:, i] - min_temp) / (max_temp - min_temp)
		labels = df1.iloc[35000:91000, -1].values
		labels = np.tile(labels[:, np.newaxis], (1, test.shape[1])).squeeze()
		labels = labels.astype(int)
		np.save(os.path.join('processed/GEC', f'labels.npy'), labels)
		np.save(os.path.join('processed/GEC', f'test.npy'), test)
		np.save(os.path.join('processed/GEC', f'train.npy'), train)
	elif dataset == 'PSM':
		dataset_folder = 'data/PSM'
		columns_to_extract = [1, 2, 3, 4, 19, 20, 21, 25]
		df1 = pd.read_csv(os.path.join(dataset_folder, 'test.csv')).iloc[:, columns_to_extract]
		df2 = pd.read_csv(os.path.join(dataset_folder, 'train.csv')).iloc[:, columns_to_extract]
		labels = pd.read_csv(os.path.join(dataset_folder, 'test_label.csv')).values[:, 1]
		train = df2.iloc[:, 1:].values
		for i in range(train.shape[0]):
			for j in range(train.shape[1]):
				if np.isnan(train[i, j]):
					train[i, j] = train[i - 1, j] if i > 0 else 0
		for i in range(train.shape[1]):
			min_temp, max_temp = np.min(train[:, i]), np.max(train[:, i])
			train[:, i] = (train[:, i] - min_temp) / (max_temp - min_temp)
		test = df1.iloc[:, 1:].values
		for i in range(test.shape[1]):
			min_temp, max_temp = np.min(test[:, i]), np.max(test[:, i])
			test[:, i] = (test[:, i] - min_temp) / (max_temp - min_temp)
		for i in range(test.shape[0]):
			for j in range(test.shape[1]):
				if np.isnan(test[i, j]):
					test[i, j] = test[i - 1, j] if i > 0 else 0
		labels = np.max(labels[:80000].reshape(-1, 20), axis=1)
		labels = np.tile(labels[:, np.newaxis], (1, 7)).squeeze()
		test = np.max(test[:80000].reshape(-1, 20, 7), axis=1)
		train = np.max(train[:80000].reshape(-1, 20, 7), axis=1)
		np.save(os.path.join('processed/PSM', f'labels.npy'), labels)
		np.save(os.path.join('processed/PSM', f'test.npy'), test)
		np.save(os.path.join('processed/PSM', f'train.npy'), train)

	elif dataset == 'Genesis':
		df1 = pd.read_csv('data/Genesis/Genesis_AnomalyLabels.csv', delimiter=',')
		df2 = pd.read_csv('data/Genesis/Genesis_normal.csv', delimiter=',')
		name_list = ['MotorData.ActCurrent', 'MotorData.ActSpeed', 'MotorData.IsAcceleration', 'MotorData.IsForce']

		test = df1.loc[:, name_list].values.astype(np.float64)
		test = np.mean(test[:16100].reshape(-1, 4, 4), axis=1)
		for i in range(test.shape[1]):
			min_temp, max_temp = np.min(test[:, i]), np.max(test[:, i])
			test[:, i] = (test[:, i] - min_temp) / (max_temp - min_temp)
		train = df2.loc[:, name_list].values.astype(np.float64)
		train = np.mean(train[:16100].reshape(-1, 4, 4), axis=1)
		for i in range(train.shape[1]):
			min_temp, max_temp = np.min(train[:, i]), np.max(train[:, i])
			train[:, i] = (train[:, i] - min_temp) / (max_temp - min_temp)
		labels = df1.loc[:, ['Label']].values
		labels = np.max(labels[:16100].reshape(-1, 4), axis=1)
		labels = np.tile(labels[:, np.newaxis], (1, 4)).squeeze()
		np.save(os.path.join('processed/Genesis', f'labels.npy'), labels)
		np.save(os.path.join('processed/Genesis', f'test.npy'), test)
		np.save(os.path.join('processed/Genesis', f'train.npy'), train)
	elif dataset == 'PUMP':
		dataset_folder = 'data/PUMP'
		df = pd.read_csv(os.path.join(dataset_folder, 'sensor.csv'))
		df.head()
		df.drop(df.columns[0], axis=1, inplace=True)
		df.describe()
		df.drop(['sensor_15'], axis=1, inplace=True)
		df.fillna(method='ffill', inplace=True)
		df.replace({'machine_status': {'RECOVERING': 1, 'BROKEN': 1, 'NORMAL': 0}}, inplace=True)
		labels = df['machine_status'].to_numpy()
		labels = np.max(labels[:3500*50].reshape(-1, 50), axis=1)
		data = df.iloc[:, 1:-1].to_numpy()
		data = np.mean(data[:220300,:].reshape(-1, 50, 51), axis=1)
		data = np.concatenate((data[:, 0:12], data[:, 37:50]), axis=1)
		for i in range(data.shape[1]):
			min_temp, max_temp = np.min(data[:, i]), np.max(data[:, i])
			data[:, i] = (data[:, i] - min_temp) / (max_temp - min_temp)
		test, train = data[:3500, :], data[3500:, :]

		labels = np.tile(labels[:3500*50, np.newaxis], (1, test.shape[1])).squeeze()
		np.save(os.path.join('processed/PUMP', f'labels.npy'), labels)
		np.save(os.path.join('processed/PUMP', f'test.npy'), test)
		np.save(os.path.join('processed/PUMP', f'train.npy'), train)
	elif dataset == 'SMD':
		dataset_folder = 'data/SMD'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
				s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
				load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
	elif dataset in ['SMAP', 'MSL']:
		dataset_folder = 'data/SMAP_MSL'
		file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
		values = pd.read_csv(file)
		values = values[values['spacecraft'] == dataset]
		filenames = values['chan_id'].values.tolist()
		for fn in filenames:
			train = np.load(f'{dataset_folder}/train/{fn}.npy')
			test = np.load(f'{dataset_folder}/test/{fn}.npy')
			train, min_a, max_a = normalize3(train)
			test, _, _ = normalize3(test, min_a, max_a)
			np.save(f'{folder}/{fn}_train.npy', train)
			np.save(f'{folder}/{fn}_test.npy', test)
			labels = np.zeros(test.shape)
			indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
			indices = indices.replace(']', '').replace('[', '').split(', ')
			indices = [int(i) for i in indices]
			for i in range(0, len(indices), 2):
				labels[indices[i]:indices[i+1], :] = 1
			np.save(f'{folder}/{fn}_labels.npy', labels)
	elif dataset == 'MBA':
		dataset_folder = 'data/MBA'
		ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
		train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
		test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
		train, test = train.values[1:,1:].astype(float), test.values[1:,1:].astype(float)
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		ls = ls.values[:,1].astype(int)
		labels = np.zeros_like(test)
		for i in range(-20, 20):
			labels[ls + i, :] = 1
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'SWaT':
		dataset_folder = 'data/SWaT'
		train = pd.read_csv(os.path.join(dataset_folder, 'SWaT_Dataset_Normal_v0.csv'))
		test = pd.read_csv(os.path.join(dataset_folder, 'SWaT_Dataset_Attack_v0.csv'))
		train = train.iloc[::20]
		test = test.iloc[::20]
		train = train.drop(columns="Timestamp")
		test = test.drop(columns="Timestamp")
		labels = test.Label

		train = train.iloc[:, :51]
		mean_train = train.mean(axis=0)
		std_train = train.std(axis=0)
		norm_train = (train-mean_train) / std_train
		norm_train = (norm_train-norm_train.min(axis=0))/(norm_train.max(axis=0)-norm_train.min(axis=0))

		test = test.iloc[:, :51]
		mean_test = test.mean(axis=0)
		std_test = test.std(axis=0)
		norm_test = (test - mean_test) / std_test
		norm_test = (norm_test-norm_test.min(axis=0))/(norm_test.max(axis=0)-norm_test.min(axis=0))


		nan_columns_train = norm_train.columns[norm_train.isna().any()].tolist()
		nan_columns_test = norm_test.columns[norm_test.isna().any()].tolist()
		common_nan_columns = list(set(nan_columns_train) | set(nan_columns_test))
		train = norm_train.drop(columns=common_nan_columns).values[:,:]
		test = norm_test.drop(columns=common_nan_columns).values[:,:]
		labels = np.tile(np.expand_dims(labels, axis=1), (1, test.shape[1]))
		labels = labels[int(len(labels) * 0.46):]
		test = test[int(len(test) * 0.46):]

		print(train.shape, test.shape, labels.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	else:
		raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")