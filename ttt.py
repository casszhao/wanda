import matplotlib.pyplot as plt
import numpy as np
import pickle
# with open('full_factcc_promptB.npy', 'rb') as f:
#     a = np.load(f)  # pruned
#     b = np.load(f)  # full


with open('fulldata_factcc_promptB_pruned_mode.pkl', 'rb') as f:
    pruned_attention_list = pickle.load(f)


with open('fulldata_factcc_promptB_full_mode.pkl', 'rb') as f:
    full_attention_list = pickle.load(f)

# pruned_attention_list = [[1, 2, 3],
#                          [2, 3, 4, 4]]  # 2 examples  the first pruned example of 3 tokens



def get_mean_std(list_of_list):

    min_length = min(len(sub_list) for sub_list in list_of_list)
    cuted_pruned_attention_list = [sub_list[:min_length] for sub_list in list_of_list]    
    mean = np.mean(cuted_pruned_attention_list, axis=0)
    std = np.std(cuted_pruned_attention_list, axis=0)

    index = np.arange(1, len(mean) + 1).tolist()

    return index, mean, std



index, mean, std = get_mean_std(pruned_attention_list)

plt.plot(index, mean, "red", label="SparseGPT")
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color="red")

index, mean, std = get_mean_std(full_attention_list)
plt.plot(index, mean, "grey", label="No Pruning")
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color="grey")

plt.legend()
plt.xlabel("Generated new token index")
plt.ylabel("Attention ratio (sum) to source input")
plt.title('Attention distribution (attention to source input) of SparseGPT and No Pruning')
plt.savefig('ttt.png')
