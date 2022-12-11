import random
import knapsack
from tqdm import tqdm

from knapsack01.BBKnapsack import BBKnapsack
import numpy as np


CAPACITY=10000
EPSILON=1000  # weights 50 on average, takes around 200 items to fill the sack
VD_MIN=1
VD_MAX=2


# Code copied from https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/
def knapSack(W, wt, val, n):
    dp = [0 for i in range(W+1)] # Making the dp array

    for i in range(1, n+1): # taking first i elements
        for w in range(W, 0, -1): # starting from back,so that we also have data of
            if wt[i-1] <= w:
                # finding the maximum value
                dp[w] = max(dp[w], dp[w-wt[i-1]]+val[i-1])
    return dp[W] # returning the maximum value of knapsack

def generate_data(vd_max=2, vd_min=1, epsilon=EPSILON, N=400, seed=5):
    '''
    Args:
        vd_max: max value density, (0 for vd_min by default)
        epsilon: weights can only come from (0, epsilon]
    '''
    np.random.seed(seed)
    random.seed(seed)

    weights = np.random.rand(N) * epsilon

    scale = vd_max - vd_min
    value_densities = np.random.rand(N) * scale + vd_min
    values = value_densities * weights

    return weights.astype(int), values.astype(int)

def generate_2d_data(vd_max=2, vd_min=1, epsilon=EPSILON, N=400, seed=5):
    '''
    Args:
        vd_max: max value density, (0 for vd_min by default)
        epsilon: weights can only come from (0, epsilon]
    '''
    np.random.seed(seed)
    random.seed(seed)

    weights_1 = np.random.rand(N) * epsilon
    weights_2 = np.random.rand(N) * epsilon

    scale = vd_max - vd_min
    value_densities = np.random.rand(N) * scale + vd_min
    values = value_densities * np.minimum(weights_1, weights_2)

    return (weights_1.astype(int), weights_2.astype(int)), values.astype(int)

def optimial_online_knapsack(values, weights, vd_min=1, vd_max=2):

    total_weight = 0
    total_value = 0

    beta = 1 / (1 + np.log(vd_max / vd_min)) * CAPACITY

    num_items = len(values)

    for i in range(num_items):

        value = values[i]
        weight = weights[i]
        density = value / weight

        # accept
        if total_weight < beta:
            threshold = vd_min
        else:
            threshold = np.exp(total_weight/beta - 1)

        if density > threshold and total_weight + weight < CAPACITY:
            total_weight += weight
            total_value += value

    return total_value, total_weight

def online_knapsack_nd(values, weights, vd_min=1, vd_max=2):

    num_dimensions = len(weights)

    total_weight = [0] * num_dimensions
    total_value = 0

    beta = 1 / (1 + np.log(vd_max / vd_min)) * CAPACITY

    num_items = len(values)

    for i in range(num_items):

        accept = True

        # consider all dimensions
        for j in range(len(weights)):

            value = values[i]
            weight = weights[j][i]
            density = value / weight

            if total_weight[j] < beta:
                threshold = vd_min
            else:
                threshold = np.exp(total_weight[j]/beta - 1)

            if density > threshold and total_weight[j] + weight < CAPACITY:
                pass
            else:
                accept = False
                break

        if accept:
            for j in range(num_dimensions):
                total_weight[j] += weight
            total_value += value

    return total_value, total_weight

if __name__ == '__main__':


    # Code for infinitesimal setting
    for i in tqdm(range(100)):

        total_online_value = 0
        total_offline_value = 0

        weights, values = generate_data(vd_max=VD_MAX, vd_min=VD_MIN, seed=i, epsilon=10000)

        online_value, online_weight = optimial_online_knapsack(values, weights)
        # print(online_value, online_weight)
        total_online_value += online_value

        offline_value = knapSack(CAPACITY, weights, values, len(values))
        # print(offline_value)
        total_offline_value += offline_value

    print('Competitive ratio:', total_offline_value / total_online_value)
    print('Online value', total_online_value / 100)

    # Code for non-infinitesimal setting
    for i in tqdm(range(100)):

        total_online_value = 0
        total_offline_value = 0

        weights, values = generate_data(vd_max=VD_MAX, vd_min=VD_MIN, seed=i, epsilon=1000)

        online_value, online_weight = optimial_online_knapsack(values, weights)
        # print(online_value, online_weight)
        total_online_value += online_value

        offline_value = knapSack(CAPACITY, weights, values, len(values))
        # print(offline_value)
        total_offline_value += offline_value

    print('Competitive ratio:', total_offline_value / total_online_value)

    # Code for multidimensional
    for i in tqdm(range(100)):

        total_online_value = 0
        total_offline_value = 0

        weights, values = generate_2d_data(vd_max=VD_MAX, vd_min=VD_MIN, seed=i, epsilon=10000)

        online_value, online_weight = online_knapsack_nd(values, weights)

    print('Online value:', online_value / 100)

