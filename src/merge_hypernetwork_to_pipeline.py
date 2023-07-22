import shared
import torch

#param layer_structure : sequence used for length, use_dropout : controlling boolean, last_layer_dropout : for compatibility check.
def parse_dropout_structure(layer_structure, use_dropout, last_layer_dropout):
    if layer_structure is None:
        layer_structure = [1, 2, 1]
    if not use_dropout:
        return [0] * len(layer_structure)
    dropout_values = [0]
    dropout_values.extend([0.3] * (len(layer_structure) - 3))
    if last_layer_dropout:
        dropout_values.append(0.3)
    else:
        dropout_values.append(0)
    dropout_values.append(0)
    return dropout_values



from torch.nn.init import normal_, xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_, zeros_
class HypernetworkModule(torch.nn.Module):
    activation_dict = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
    }
    def __init__(self, dim, state_dict=None, layer_structure=None, activation_func=None, weight_init='Normal',
                 add_layer_norm=False, activate_output=False, dropout_structure=None):
        super().__init__()

        self.multiplier = 1.0

        assert layer_structure is not None, "layer_structure must not be None"
        assert layer_structure[0] == 1, "Multiplier Sequence should start with size 1!"
        assert layer_structure[-1] == 1, "Multiplier Sequence should end with size 1!"

        linears = []
        for i in range(len(layer_structure) - 1):

            # Add a fully-connected layer
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i+1])))

            # Add an activation func except last layer
            if activation_func == "linear" or activation_func is None or (i >= len(layer_structure) - 2 and not activate_output):
                pass
            elif activation_func in self.activation_dict:
                linears.append(self.activation_dict[activation_func]())
            else:
                raise RuntimeError(f'hypernetwork uses an unsupported activation function: {activation_func}')

            # Add layer normalization
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i+1])))

            # Everything should be now parsed into dropout structure, and applied here.
            # Since we only have dropouts after layers, dropout structure should start with 0 and end with 0.
            if dropout_structure is not None and dropout_structure[i+1] > 0:
                assert 0 < dropout_structure[i+1] < 1, "Dropout probability should be 0 or float between 0 and 1!"
                linears.append(torch.nn.Dropout(p=dropout_structure[i+1]))
            # Code explanation : [1, 2, 1] -> dropout is missing when last_layer_dropout is false. [1, 2, 2, 1] -> [0, 0.3, 0, 0], when its True, [0, 0.3, 0.3, 0].

        self.linear = torch.nn.Sequential(*linears)

        if state_dict is not None:
            self.fix_old_state_dict(state_dict)
            self.load_state_dict(state_dict)
        else:
            for layer in self.linear:
                if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                    w, b = layer.weight.data, layer.bias.data
                    if weight_init == "Normal" or type(layer) == torch.nn.LayerNorm:
                        normal_(w, mean=0.0, std=0.01)
                        normal_(b, mean=0.0, std=0)
                    elif weight_init == 'XavierUniform':
                        xavier_uniform_(w)
                        zeros_(b)
                    elif weight_init == 'XavierNormal':
                        xavier_normal_(w)
                        zeros_(b)
                    elif weight_init == 'KaimingUniform':
                        kaiming_uniform_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
                        zeros_(b)
                    elif weight_init == 'KaimingNormal':
                        kaiming_normal_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
                        zeros_(b)
                    else:
                        raise KeyError(f"Key {weight_init} is not defined as initialization!")
        self.to(shared.device)

    def fix_old_state_dict(self, state_dict):
        changes = {
            'linear1.bias': 'linear.0.bias',
            'linear1.weight': 'linear.0.weight',
            'linear2.bias': 'linear.1.bias',
            'linear2.weight': 'linear.1.weight',
        }

        for fr, to in changes.items():
            x = state_dict.get(fr, None)
            if x is None:
                continue

            del state_dict[fr]
            state_dict[to] = x

    def forward(self, x):
        return x + self.linear(x) * (self.multiplier if not self.training else 1)


def merge_hypernetwork_to_pipeline(text_encoder, unet, hypernetwork_state_dict, alpha):
    for size, sd in hypernetwork_state_dict.items():
        if type(size) is not int:
            print(size, " = ", sd)
        else:
            for i in range(len(sd)):
                for key, val in sd[i].items():
                    keys = key.split(".")

                    if size == 320:
                        block_index = 0
                    elif size == 640:
                        block_index = 1
                    elif size == 768:
                        block_index = 2
                    elif size == 1280:
                        block_index = 3

                    if i == 0:
                        block_name = "down_blocks"
                    elif i == 1:
                        block_name = "up_blocks"

                    index = int(keys[1])
                    weight_or_bias = keys[2]

                    hypernetwork_tensor = val

                    blocks = unet.__getattr__(block_name)
                    block = blocks.__getattr__(str(block_index))

                    print(block.attentions[0].transformer_blocks[0].attn1.to_k.weight)

    #print(pipeline.unet)

    return

    layer_structure = state_dict.get('layer_structure', [1, 2, 1])
    optional_info = state_dict.get('optional_info', None)
    activation_func = state_dict.get('activation_func', None)
    weight_init = state_dict.get('weight_initialization', 'Normal')
    add_layer_norm = state_dict.get('is_layer_norm', False)
    dropout_structure = state_dict.get('dropout_structure', None)
    use_dropout = True if dropout_structure is not None and any(dropout_structure) else state_dict.get('use_dropout', False)
    activate_output = state_dict.get('activate_output', True)
    last_layer_dropout = state_dict.get('last_layer_dropout', False)

    if dropout_structure is None:
        dropout_structure = parse_dropout_structure(layer_structure, use_dropout, last_layer_dropout)

    layers = {}

    for size, sd in state_dict.items():
        if type(size) == int:
            layers[size] = (
                HypernetworkModule(size, sd[0], layer_structure, activation_func, weight_init,
                                    add_layer_norm, activate_output, dropout_structure),
                HypernetworkModule(size, sd[1], layer_structure, activation_func, weight_init,
                                    add_layer_norm, activate_output, dropout_structure),
            )

    name = state_dict.get('name', checkpoint_path)
    step = state_dict.get('step', 0)
    sd_checkpoint = state_dict.get('sd_checkpoint', None)
    sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)
    
    for layers in layers.values():
        for layer in layers:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False