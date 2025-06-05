import gymnasium.spaces
import pandas as pd
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt


def dict_to_tensor(dic, ksp=None):
    """
    takes a gymnasium.spaces.Dict({k:gymnasium.spaces.Box}) and converts it to tensor
    Args:
        dic: dictionary of string to array
        ksp: keys, shape, partition (object returned by this function)
            if not specified, makes this from dic
    Returns:
        (tensor, (keys,shapes,partition))
        tensor is a 1d tensor of everything in dic flattened and concatenated
        keys is a tuple of keys (same as input if specified)
        shapes is a tuple of shapes (same as input if specified), where dic[keys[i]].shape=shapes[i]
        partition is a tuple of indices, where tensor[partition[i]:partition[i+1]] = dic[keys[i]]
    """
    if ksp is None:
        keys = tuple(sorted(list(dic.keys())))
        shapes = tuple(dic[key].shape for key in keys)
        partition = tuple(np.cumsum([0] + [np.prod(shape) for shape in shapes]))
    else:
        keys, shapes, partition = ksp
    tense = torch.concatenate(
        [
            dic[key].flatten() if type(dic[key]) == torch.Tensor
            else torch.tensor(dic[key].flatten(), dtype=torch.float)
            for key in keys
        ]
    )

    return tense, (keys, shapes, partition)


def tensor_to_dict(tensor, ksp):
    """
    inverse of dict_to_tensor
    Args:
        tensor: 1D flattened tensor
        ksp: (keys, shapes, partition) returned by dict_to_tensor
    Returns:
        dic that was input into dict_to_tensor
    """
    keys, shapes, partition = ksp
    dic = {
        key: tensor[partition[i]:partition[i + 1]].reshape(shapes[i])
        for i, key in enumerate(keys)
    }
    return dic


class DicWrapper(torch.nn.Module):
    """
    network is a torch module that takes in a dictionary
    DicWrapper(network) takes in a tensor such that
        DicWrapper(network)(dict_to_tensor(dic)) = network(dic)
    """

    def __init__(self, network: torch.nn.Module, ksp):
        super(DicWrapper, self).__init__()
        self.network = network
        self.ksp = ksp

    def forward(self, tense):
        return self.network.forward(tensor_to_dict(tensor=tense, ksp=self.ksp))


def shap_val(model,background):
    explainer = shap.DeepExplainer(model, background_tensor)


if __name__ == '__main__':
    test = {
        'a': np.random.rand(*(1 + np.arange(7))),
        'b': np.random.rand(4),
        'c': np.random.rand(69, 420),
    }
    output = dict_to_tensor(dic=test)
    tensor = output[0]
    test2 = tensor_to_dict(*output)
    assert all(torch.equal(torch.tensor(test[k],dtype=torch.float), test2[k]) for k in test)
    for k in test:
        assert test[k].shape == test2[k].shape


    # define architecture or load model here
    class DNeuralNet(torch.nn.Module):
        def __init__(self):
            super(DNeuralNet, self).__init__()
            self.fc1 = torch.nn.Linear(360, 8)
            self.fc2 = torch.nn.Linear(8, 1)

        def forward(self, x):
            x = x['test']
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    model = DNeuralNet()


    example_input = {'test': torch.rand(1, 360)}
    tense, ksp = dict_to_tensor(example_input)
    wrapped_model = DicWrapper(model, ksp=ksp)
    print([param.shape for param in model.parameters()])
    print([param.shape for param in wrapped_model.parameters()])
    print(wrapped_model.forward(tense=tense),model.forward(example_input))

quit()

plt.close('all')

model = NeuralNet()
model.load_state_dict(
    torch.load("real_bee_model-X-1.pth", map_location=torch.device('cpu')))  # load model (u can/should use gpu)
model.eval()

csv_file = "full_simulation_data/150_0_0_0_0_output-simple-for-shap_network_inputs.csv"  # load input data
df = pd.read_csv(csv_file, header=None)

total_samples = df.shape[0]
start_index = (total_samples - 2000)//2
middle_data = df.iloc[start_index: start_index + 2000].values.astype(np.float32)
assert middle_data.shape[1] == 360, "CSV must have 360 features per row."
print("Middle data shape:", middle_data.shape)

# Use the full 2000 samples as the explanation dataset(can use the whole data instead of sampling it)
explanation_data = middle_data  # you have to set the entire dataset that is supposed to be the background for the Shapley values
num_explanations = explanation_data.shape[0]

# Randomly pick 200 samples from the 2000 for background data
np.random.seed(42)
background_indices = np.random.choice(num_explanations, size=200,
                                      replace=False)  # this is to build a background dataset for the shapley value to build a baseline on
background_data = explanation_data[background_indices]
print("Background data shape:", background_data.shape)

# Convert data to torch.Tensors for DeepExplainer
background_tensor = torch.tensor(background_data, dtype=torch.float32)
explanation_tensor = torch.tensor(explanation_data, dtype=torch.float32)
print("Explanation tensor shape:", explanation_tensor.shape)

# Initialize SHAP DeepExplainer
explainer = shap.DeepExplainer(model, background_tensor)

all_shap = []
for i in range(num_explanations):  # this is to generate the shapley values for each sample in the dataset
    sample = explanation_tensor[i: i + 1]  # this is just for my models dimensional requirement
    shap_values_list = explainer.shap_values(sample, check_additivity=False)
    all_shap.append(shap_values_list[0].squeeze())

# Stack into a (2000, 360) array
shap_values = np.stack(all_shap, axis=0)
print("SHAP values shape:", shap_values.shape)

# expected_value = explainer.expected_value[0]
# print("Expected value:", expected_value)

# now you have the shapley value for all the inputs, so u can just map them however you want.
# here i have shown a very simple example, but you can combine segments of an image or build entire attention maps by combining the contributions of sections (i have an example of this commented out below)

sample_indices = np.arange(num_explanations)
feature_0 = shap_values[:, 0]
feature_90 = shap_values[:, 90]
feature_180 = shap_values[:, 180]

plt.figure(figsize=(10, 6))
plt.plot(sample_indices, feature_0, label="Feature 0")
plt.plot(sample_indices, feature_90, label="Feature 90")
plt.plot(sample_indices, feature_180, label="Feature 180")
plt.xlabel("Sample Index (Time)")
plt.ylabel("SHAP Value")
plt.title("SHAP Values for Features 0, 90, and 180 Over 2000 Samples")
plt.legend()
plt.show()

#
# # Current time:
# current_outer = np.sum(shap_values[:, 0:45], axis=1) + np.sum(shap_values[:, 135:180], axis=1)
# current_center = np.sum(shap_values[:, 45:135], axis=1)
#
# # Old time:
# old_outer = np.sum(shap_values[:, 180:225], axis=1) + np.sum(shap_values[:, 315:360], axis=1)
# old_center = np.sum(shap_values[:, 225:315], axis=1)
#
# window_size = 15  # Must be odd; adjust as needed
# poly_order = 3
#
# # This smoothens the outputs
# current_outer_smooth = savgol_filter(current_outer, window_size, poly_order)
# current_center_smooth = savgol_filter(current_center, window_size, poly_order)
# old_outer_smooth = savgol_filter(old_outer, window_size, poly_order)
# old_center_smooth = savgol_filter(old_center, window_size, poly_order)
#
# # Loading the positions so that I can plot the shapley values along the X position.
# # U have to do something similar for the Velocity, Acceleration and SHAP that will show use how these change across the tunnel length/trajectory
# positions_file = "full_simulation_data/150_0_0_0_0_output-simple-for-shap_positions.csv"
# positions_df = pd.read_csv(positions_file, header=0)
#
# x_positions = positions_df.iloc[:, 0].astype(float).values
#
# # Normalize the true x positions:
# x_min = x_positions.min()
# x_max = x_positions.max()
# normalized_x = 2000 * (x_positions - x_min) / (x_max - x_min)
#
# print(f"Original x positions range: {x_min} to {x_max}")
# print(f"Normalized x positions range: {normalized_x.min()} to {normalized_x.max()}")
#
# if len(normalized_x) != num_explanations:
#     print(
#         f"Warning: Number of x positions ({len(normalized_x)}) does not match number of samples ({num_explanations}). Using first {num_explanations} positions.")
#     normalized_x = normalized_x[:num_explanations]
#
# print("Normalized x_positions shape:", normalized_x.shape)

# plt.figure(figsize=(12, 8))
# plt.plot(normalized_x, current_outer_smooth, label="Current Outer (Sum of left/right outer)")
# plt.plot(normalized_x, current_center_smooth, label="Current Center (Middle 90)")
# plt.plot(normalized_x, old_outer_smooth, label="Old Outer (Sum of left/right outer)")
# plt.plot(normalized_x, old_center_smooth, label="Old Center (Middle 90)")
#
# plt.xlabel("X Position (Normalized)")
# plt.ylabel("Sum of SHAP Values (Smoothed)")
# plt.title("SHAP Contributions: Outer vs Center for Current (t) and Old (t-1)")
# plt.xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
# plt.legend(prop={'size': 12})
# plt.show()
