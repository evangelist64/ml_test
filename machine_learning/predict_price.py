import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

def get_data(file_name):
	data = pd.read_csv(file_name)
	X_parameter = []
	Y_parameter = []
	for single_square_feet ,single_price_value in zip(data['square_feet'],data['price']):
		X_parameter.append([float(single_square_feet)])
		Y_parameter.append(float(single_price_value))
	return X_parameter,Y_parameter
