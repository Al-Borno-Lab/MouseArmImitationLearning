import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from enum import Enum

#==========================================================================
# Recurrence
#==========================================================================

class ReccurentLayerType(Enum):
    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"


class RecurrentLayers(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length, num_envs):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.num_envs = num_envs

    def set_training(self, enabled: bool):
        self.training = enabled

    def get_specification(self):
        # batch size (N) is the number of envs
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [(self.num_layers, self.num_envs, self.hidden_size)],
            }
        }  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs):
        observations = inputs["observations"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # training
        if self.training:
            rnn_input = observations.view(
                -1, self.sequence_length, observations.shape[-1]
            )  # (N, L, Hin): N=batch_size, L=sequence_length

            hidden_states = hidden_states.view(
                self.num_layers, -1, self.sequence_length, hidden_states.shape[-1]
            )  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.rnn(rnn_input[:, i0:i1, :], hidden_states)
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = observations.view(-1, 1, observations.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return rnn_output, {"rnn": [hidden_states]}


class RNN(RecurrentLayers):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length, num_envs):
        super().__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, sequence_length=sequence_length, num_envs=num_envs)
        self.rnn = nn.RNN(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )  # batch_first -> (batch, sequence, features)
    
class GRU(RecurrentLayers):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length, num_envs):
        super().__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, sequence_length=sequence_length, num_envs=num_envs)
        self.gru = nn.GRU(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )  # batch_first -> (batch, sequence, features)


class LSTM(RecurrentLayers):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length, num_envs):
        super().__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, sequence_length=sequence_length, num_envs=num_envs)
        self.lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )  # batch_first -> (batch, sequence, features)

    def get_specification(self):
        # batch size (N) is the number of envs
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),  # hidden states (D ∗ num_layers, N, Hout)
                    (self.num_layers, self.num_envs, self.hidden_size),
                ],
            }
        }  # cell states (D ∗ num_layers, N, Hcell)
    
    def compute(self, inputs):
        observations = inputs["observations"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            rnn_input = observations.view(
                -1, self.sequence_length, observations.shape[-1]
            )  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(
                self.num_layers, -1, self.sequence_length, hidden_states.shape[-1]
            )  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(
                self.num_layers, -1, self.sequence_length, cell_states.shape[-1]
            )  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(
                        rnn_input[:, i0:i1, :], (hidden_states, cell_states)
                    )
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    cell_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = observations.view(-1, 1, observations.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return rnn_output, {"rnn": [rnn_states[0], rnn_states[1]]}
    
#==========================================================================
# Model
#==========================================================================

# define the shared model
class SharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        num_envs,

        clip_actions=True,
        clip_mean_actions=False,
        clip_log_std=True,   # keep std numerically sane
        min_log_std=-20,     # std can get extremely tiny, almost deterministic
        max_log_std=2,       # std can get large, but not insane
        reduction="sum",     # combine per-action log_probs into one PPO log_prob
        
        sequence_length=50,

        rnn_type=ReccurentLayerType.LSTM,
        rnn_hidden_size=256,
        rnn_layers=1,
        policy_hidden_size=128,
        policy_layers=2,
        value_hidden_size=128,
        value_layers=2,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=clip_mean_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            role="policy",
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")
        
        # shared layers (backbone)
        self.reccurent_layers: RecurrentLayers
        if rnn_type == ReccurentLayerType.RNN:
            self.reccurent_layers = RNN(
                input_size=self.num_observations,
                hidden_size=rnn_hidden_size, 
                num_layers=rnn_layers, 
                sequence_length=sequence_length, 
                num_envs=num_envs
            )
        elif rnn_type == ReccurentLayerType.GRU:
            self.reccurent_layers = GRU(
                input_size=self.num_observations,
                hidden_size=rnn_hidden_size, 
                num_layers=rnn_layers, 
                sequence_length=sequence_length, 
                num_envs=num_envs
            )
        elif rnn_type == ReccurentLayerType.LSTM:
            self.reccurent_layers = LSTM(
                input_size=self.num_observations,
                hidden_size=rnn_hidden_size, 
                num_layers=rnn_layers, 
                sequence_length=sequence_length, 
                num_envs=num_envs
            )
        else:
            raise ValueError(
                f"Unsupported rnn_type: {rnn_type!r}. "
                f"Expected one of: "
                f"{ReccurentLayerType.RNN}, "
                f"{ReccurentLayerType.GRU}, "
                f"{ReccurentLayerType.LSTM}."
            )
        hidden_size = self.reccurent_layers.hidden_size

        # separated layers ("policy" head)
        policy_layers_list = []

        in_features = hidden_size
        for _ in range(policy_layers):
            policy_layers_list.append(nn.Linear(in_features, policy_hidden_size))
            policy_layers_list.append(nn.Tanh())
            in_features = policy_hidden_size

        policy_layers_list.append(nn.Linear(in_features, self.num_actions))
        policy_layers_list.append(nn.Tanh())

        self.mean_layer = nn.Sequential(*policy_layers_list)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))


        # separated layers ("value" head)
        value_layers_list = []

        in_features = hidden_size
        for _ in range(value_layers):
            value_layers_list.append(nn.Linear(in_features, value_hidden_size))
            value_layers_list.append(nn.Tanh())
            in_features = value_hidden_size

        value_layers_list.append(nn.Linear(in_features, 1))

        self.value_layer = nn.Sequential(*value_layers_list)


        # action scaling values for mapping tanh output [-1, 1] -> action_space [low, high]
        action_low = torch.as_tensor(
            self.action_space.low,
            dtype=torch.float64,
            device=self.device,
        )

        action_high = torch.as_tensor(
            self.action_space.high,
            dtype=torch.float64,
            device=self.device,
        )

        self.action_center = (action_high + action_low) / 2.0
        self.action_scale = (action_high - action_low) / 2.0

    def enable_training_mode(self, enabled = True):
        output = super().enable_training_mode(enabled)
        self.reccurent_layers.set_training(enabled)
        return output

    # override the .act(...) method to disambiguate its call
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role=role)

    def get_specification(self):
        return self.reccurent_layers.get_specification()

    # forward the input to compute model output according to the specified role
    def compute(self, inputs, role):
        if role == "policy":
            # save shared layers/network output to perform a single forward-pass
            self._shared_output, data = self.reccurent_layers.compute(inputs)
            data["log_std"] = self.log_std_parameter

            raw_mean = self.mean_layer(self._shared_output)  # final tanh, range [-1, 1]
            mean = self.action_center + raw_mean * self.action_scale
            
            return mean, data
        elif role == "value":
            # use saved shared layers/network output to perform a single forward-pass, if it was saved
            if self._shared_output is None:
                shared_output, data = self.reccurent_layers.compute(inputs)
            else:
                shared_output = self._shared_output
            self._shared_output = (
                None  # reset saved shared output to prevent the use of erroneous data in subsequent steps
            )
            return self.value_layer(shared_output), {}
