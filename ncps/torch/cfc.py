# Copyright 2022 Mathias Lechner and Ramin Hasani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch import nn
from typing import Optional, Union
import ncps
from . import CfCCell, WiredCfCCell
from .lstm import LSTMCell


class CfC(nn.Module):
    def __init__(
        self,
        input_size: Union[int, ncps.wirings.Wiring],
        units,
        proj_size: Optional[int] = None,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: Optional[int] = None,
        backbone_layers: Optional[int] = None,
        backbone_dropout: Optional[int] = None,
    ):
        super(CfC, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.proj_size = proj_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        if isinstance(units, ncps.wirings.Wiring) or isinstance(units, ncps.wirings.wiringsRevised.WiringRevised):
            self.wired_mode = True
            if backbone_units is not None:
                raise ValueError(f"Cannot use backbone_units in wired mode")
            if backbone_layers is not None:
                raise ValueError(f"Cannot use backbone_layers in wired mode")
            if backbone_dropout is not None:
                raise ValueError(f"Cannot use backbone_dropout in wired mode")
            self.wiring = units
            self.state_size = self.wiring.units
            self.output_size = self.wiring.output_dim if self.wiring.output_dim is not None else self.state_size
            self.rnn_cell = WiredCfCCell(
                input_size,
                self.wiring_or_units,
                mode,
            )
        else:
            self.wired_false = True
            backbone_units = 128 if backbone_units is None else backbone_units
            backbone_layers = 1 if backbone_layers is None else backbone_layers
            backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout
            self.state_size = units
            self.output_size = self.state_size
            self.rnn_cell = CfCCell(
                input_size,
                self.wiring_or_units,
                mode,
                activation,
                backbone_units,
                backbone_layers,
                backbone_dropout,
            )
        self.use_mixed = mixed_memory
        if self.use_mixed:
            self.lstm = LSTMCell(input_size, self.state_size)

        if proj_size is None:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.output_size, self.proj_size)

    def forward(self, input, hx=None, timespans=None):
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
            c_state = torch.zeros((batch_size, self.state_size), device=device) if self.use_mixed else None
        else:
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError(
                    "Running a CfC with mixed_memory=True, requires a tuple (h0,c0) to be passed as state (got torch.Tensor instead)"
                )
            h_state, c_state = hx if self.use_mixed else (hx, None)
            if is_batched:
                if h_state.dim() != 2:
                    msg = f"For batched 2-D input, hx and cx should also be 2-D but got ({h_state.dim()}-D) tensor"
                    raise RuntimeError(msg)
            else:
                if h_state.dim() != 1:
                    msg = f"For unbatched 1-D input, hx and cx should also be 1-D but got ({h_state.dim()}-D) tensor"
                    raise RuntimeError(msg)
                h_state = h_state.unsqueeze(0)
                c_state = c_state.unsqueeze(0) if c_state is not None else None

        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()

            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if self.return_sequences:
                output_sequence.append(self.fc(h_out))

        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = self.fc(h_out)
        hx = (h_state, c_state) if self.use_mixed else h_state

        if not is_batched:
            readout = readout.squeeze(batch_dim)
            hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]

        return readout, hx
