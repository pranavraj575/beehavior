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
        dic: dictionary of string to array, or list of these
        ksp: keys, shape, partition (object returned by this function)
            if not specified, makes this from dic
    Returns:
        (tensor, (keys,shapes,partition))
        tensor is a 1d tensor of everything in dic flattened and concatenated
            if dic is a list of dicts, then tensor is 2d, with first dim the size of list
        keys is a tuple of keys (same as input if specified)
        shapes is a tuple of shapes (same as input if specified), where dic[keys[i]].shape=shapes[i]
        partition is a tuple of indices, where tensor[partition[i]:partition[i+1]] = dic[keys[i]]
    """
    if type(dic) != dict:
        if ksp is None:
            _, ksp = dict_to_tensor(dic=dic[0], ksp=ksp)
        return torch.stack([dict_to_tensor(dic=d, ksp=ksp)[0] for d in dic], dim=0), ksp

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
        tensor: 1D flattened tensor or 2D stacked tensor
        ksp: (keys, shapes, partition) returned by dict_to_tensor
    Returns:
        dict (or dict list) that was input into dict_to_tensor
    """
    if len(tensor.shape) == 2:  # split the first dimension
        return [tensor_to_dict(tensor=ten, ksp=ksp) for ten in tensor]

    keys, shapes, partition = ksp
    dic = {
        key: tensor[partition[i]:partition[i + 1]].reshape(shapes[i])
        for i, key in enumerate(keys)
    }
    return dic


class GymWrapper(torch.nn.Module):
    """
    network is a gym network, network.forward outputs a tuple
        this returns the ith element of the tuple
    """

    def __init__(self,
                 network: torch.nn.Module,
                 i=0,
                 model_call_kwargs=None,
                 ):
        super(GymWrapper, self).__init__()
        self.network = network
        self.i = i
        if model_call_kwargs is None:
            model_call_kwargs = dict()
        self.model_call_kwargs = model_call_kwargs

    def forward(self, tense):
        mod_out = self.network.forward(tense, **self.model_call_kwargs)[self.i]
        if len(mod_out.shape) == 1:
            mod_out = torch.unsqueeze(mod_out, dim=0)
        return mod_out


class DicWrapper(torch.nn.Module):
    """
    network is a torch module that takes in a dictionary
    DicWrapper(network) takes in a tensor such that
        DicWrapper(network)(dict_to_tensor(dic)) = network(dic)

    if proc_model_output is not the identity, then
        DicWrapper(network)(dict_to_tensor(dic)) = proc_model_output(network(dic))
    """

    def __init__(self,
                 network: torch.nn.Module,
                 ksp,
                 proc_model_output=lambda x: x,
                 model_call_kwargs=None,
                 ):
        """
        Args:
            network:
            ksp:
            proc_model_output:
            model_call_kwargs: keyword args to pass: tensor x -> network.forward(x,**model_call_kwargs)
        """
        super(DicWrapper, self).__init__()
        self.network = network
        self.ksp = ksp
        self.proc_model_output = proc_model_output
        if model_call_kwargs is None:
            model_call_kwargs = dict()
        self.model_call_kwargs = model_call_kwargs

    def forward(self, tense):
        revert = tensor_to_dict(tensor=tense, ksp=self.ksp)
        if type(revert) == dict:
            return self.network.forward(revert, **self.model_call_kwargs)
        else:
            if self.proc_model_output is not None:
                return torch.stack([
                    self.proc_model_output(self.network.forward(dic, **self.model_call_kwargs))
                    for dic in revert],
                    dim=0,
                )
            else:
                return torch.stack([
                    self.network.forward(dic, **self.model_call_kwargs)
                    for dic in revert],
                    dim=0,
                )


def shap_val(model, explanation_data, baseline, ):
    """
    returns shap values for explanation_data based on baseline
    Args:
        model: nn.Module that goes from tensor -> tensor
        explanation_data: batch of tensor input to explain, shaped (N,*)
        baseline: baseline of tensor input for use, shaped (M,*)
    Returns:
        list of explanations
            each explanation is a list of shap values (shaped (*)), one for each output dim
            if model is a (shape_in -> k vector) map,
                each explanation will be a list (length k), where each element is shaped shape_in
    """
    # baseline should be about 10% sample from explanation
    explainer = shap.DeepExplainer(model, baseline, )
    explanations = []
    for i in range(len(explanation_data)):
        expln = explainer.shap_values(explanation_data[i:i + 1], check_additivity=False)
        explanations.append(expln)
    return explanations


if __name__ == '__main__':
    n = 100
    test = {
        'a': np.random.rand(*(1 + np.arange(7))),
        'b': np.random.rand(4),
        'c': np.random.rand(69, 420),
    }
    output = dict_to_tensor(dic=test)
    tensor = output[0]
    test2 = tensor_to_dict(*output)
    assert all(torch.equal(torch.tensor(test[k], dtype=torch.float), test2[k]) for k in test)
    for k in test:
        assert test[k].shape == test2[k].shape


    class NeuralNet(torch.nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()
            self.fc1 = torch.nn.Linear(360, 8)
            self.test_lyr = torch.nn.Tanh()
            self.fc2 = torch.nn.Linear(8, 1)

        def forward(self, x):
            x = self.fc1(x)
            # x = torch.relu(x)
            x = self.test_lyr(x)
            x = self.fc2(x)
            return x


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
    print(wrapped_model.forward(tense=tense), model.forward(example_input))

    torch.random.manual_seed(69)
    explanation_data = torch.rand(n, 360)
    model = NeuralNet()
    shap_val(model=model,
             explanation_data=explanation_data,
             baseline=explanation_data[torch.randint(0, len(explanation_data), (200,))],
             )

    import gymnasium as gym
    from stable_baselines3 import PPO

    env = gym.make('MountainCarContinuous-v0')
    model = PPO(policy='MlpPolicy',
                env=env,
                device='cpu'
                )
    explanation_data = torch.concatenate([model.policy.obs_to_tensor(env.observation_space.sample())[0]
                                          for _ in range(n)],
                                         dim=0)

    shap_val(model=GymWrapper(model.policy, ),
             explanation_data=explanation_data,
             baseline=explanation_data,
             )

    from beehavior.envs.test import Test2, TestNN2

    env = Test2()
    obs = env.reset()[0]
    policy_kwargs = dict(
        features_extractor_class=TestNN2,
    )
    model = PPO(policy='MlpPolicy',
                env=env,
                policy_kwargs=policy_kwargs,
                device='cpu',
                )

    explanation_data = torch.concatenate([model.policy.obs_to_tensor(env.observation_space.sample())[0]
                                          for _ in range(n)], dim=0)

    print(explanation_data.shape)
    print(model.predict(obs))
    wrapped_model = GymWrapper(network=model.policy)
    print(wrapped_model.forward(explanation_data)[0].shape)
    shap_val(model=wrapped_model,
             explanation_data=explanation_data,
             baseline=explanation_data,
             )

    from beehavior.envs.test import Test, TestNN

    env = Test()
    obs = env.reset()[0]
    policy_kwargs = dict(
        features_extractor_class=TestNN,
    )
    model = PPO(policy='MultiInputPolicy',
                env=env,
                policy_kwargs=policy_kwargs,
                device='cpu',
                )

    explanation_data = [model.policy.obs_to_tensor(env.observation_space.sample())[0]
                        for _ in range(n)]
    explanation_data_tensor, ksp = dict_to_tensor(explanation_data)
    print(explanation_data_tensor.shape)
    print(model.predict(obs))
    wrapped_model = DicWrapper(network=model.policy,
                               ksp=ksp,
                               model_call_kwargs={'deterministic': True},
                               proc_model_output=lambda x: x[0].flatten(),
                               )
    print(wrapped_model.forward(explanation_data_tensor).shape)
    shap_val(model=wrapped_model,
             explanation_data=explanation_data_tensor,
             baseline=explanation_data_tensor,
             )

    quit()

    plt.close('all')

    model = NeuralNet()

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
