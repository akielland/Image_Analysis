import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
torch.manual_seed(123)  # Set the seed for PyTorch

class ImageCaptionModel(nn.Module):
    def __init__(self, config: dict):
        """
        This is the main module class for the image captioning network
        :param config: dictionary holding neural network configuration
        """
        super(ImageCaptionModel, self).__init__()
        # Store config values as instance variables
        self.vocabulary_size = config['vocabulary_size']  # 10 000 words
        self.embedding_size = config['embedding_size']   # 300 element in a vector representing different feature of the word's meaning
        self.number_of_cnn_features = config['number_of_cnn_features']  #
        self.hidden_state_sizes = config['hidden_state_sizes']
        self.num_rnn_layers = config['num_rnn_layers']
        self.cell_type = config['cellType']

        # Create the network layers
        self.embedding_layer = nn.Embedding(self.vocabulary_size, self.embedding_size)
        # TODO: The output layer (final layer) is a linear layer. What should be the size (dimensions) of its output?
        #         Replace None with a linear layer with correct output size
        self.output_layer = nn.Linear(self.hidden_state_sizes, self.vocabulary_size) # nn.Linear(self.hidden_state_sizes, )
        self.nn_map_size = 512  # The output size for the image features after the processing via self.inputLayer
        # TODO: Check the task description and replace None with the correct input layer
        self.input_layer = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(self.number_of_cnn_features, self.nn_map_size),
            nn.LeakyReLU()
        )

        self.simplified_rnn = False
        
        if self.simplified_rnn:
            # Simplified one layer RNN is used for task 1 only.
            if self.cell_type != 'RNN':
                raise ValueError('config["cellType"] must be "RNN" when self.simplified_rnn has been set to True.'
                                 'It is ', self.cell_type, 'instead.')

            if self.num_rnn_layers != 1:
                raise ValueError('config["num_rnn_layers"] must be 1 for simplified RNN.'
                                 'It is', self.num_rnn_layers, 'instead.')

            self.rnn = RNNOneLayerSimplified(input_size=self.embedding_size + self.nn_map_size,
                                             hidden_state_size=self.hidden_state_sizes)
        else:
            self.rnn = RNN(input_size=self.embedding_size + self.nn_map_size,
                           hidden_state_size=self.hidden_state_sizes,
                           num_rnn_layers=self.num_rnn_layers,
                           cell_type=self.cell_type)

    def forward(self, cnn_features, x_tokens, is_train: bool, current_hidden_state=None) -> tuple:
        """
        :param cnn_features: Features from the CNN network, shape[batch_size, number_of_cnn_features]
        :param x_tokens: Shape[batch_size, truncated_backprop_length]
        :param is_train: A flag used to select whether or not to use estimated token as input
        :param current_hidden_state: If not None, it should be passed into the rnn module. It's shape should be
                                    [num_rnn_layers, batch_size, hidden_state_sizes].
        :return: logits of shape [batch_size, truncated_backprop_length, vocabulary_size] and new current_hidden_state
                of size [num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # HINT: For task 4, you might need to do self.input_layer(torch.transpose(cnn_features, 1, 2))
        processed_cnn_features = self.input_layer(cnn_features)

        if current_hidden_state is None:
            # TODO: Initialize initial_hidden_state with correct dimensions depending on the cell type.
            # The shape of the hidden state here should be [num_rnn_layers, batch_size, hidden_state_sizes].
            # Remember that each rnn cell needs its own initial state.
            batch_size = cnn_features.size(0)
            initial_hidden_state = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_state_sizes)
            initial_hidden_state = initial_hidden_state.to(cnn_features.device)

        else:
            initial_hidden_state = current_hidden_state

        # Call self.rnn to get the "logits" and the new hidden state
        logits, hidden_state = self.rnn(x_tokens, processed_cnn_features, initial_hidden_state,
                                        self.output_layer, self.embedding_layer, is_train)

        return logits, hidden_state

######################################################################################################################


class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, cell_type='GRU'):
        """
        :param input_size: Size of the embeddings
        :param hidden_state_size: Number of units in the RNN cells (will be equal for all RNN layers)
        :param num_rnn_layers: Number of stacked RNN layers
        :param cell_type: The type cell to use like vanilla RNN, GRU or GRU.
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers = num_rnn_layers
        self.cell_type = cell_type

        # TODO: len(input_size_list) == num_rnn_layers.
        #  input_size_list[i] should contain the input size for layer i.
        # used later when populating self.cells to give correct input
        # hidden state size is the same for all layers; as defined by hidden_state_size
        input_size_list = [input_size] + [hidden_state_size] * (num_rnn_layers - 1)

        # TODO: Create a list (self.cells) of type "nn.ModuleList" and populate it with cells of type
        #       "self.cell_type" - depending on the number of RNN layers
        rnn_cells = []    # list to store the RNN cell instances
        # loop through RNN layers (just 2 layers here; self.cells will have 2 cells in the end)
        for i in range(num_rnn_layers):
            if cell_type == 'GRU':
                rnn_cell_class = GRUCell
            elif cell_type == 'LSTM':
                rnn_cell_class = LSTMCell

            # Create an instance of the RNN cells with the appropriate input size and hidden state size
            # cell input size is determined by input_size_list[i]: 0=input_size; 1=hidden_state_size of prev layer
            rnn_cell_instance = rnn_cell_class(hidden_state_size, input_size_list[i])
            rnn_cells.append(rnn_cell_instance)

        # wrap list of RNN cells in a nn.ModuleList and assign to self.cells
        self.cells = nn.ModuleList(rnn_cells)

    def forward(self, tokens, processed_cnn_features, initial_hidden_state, output_layer: nn.Linear,
                embedding_layer: nn.Embedding, is_train=True) -> tuple:
        """
        :param tokens: Words and chars that are to be used as inputs for the RNN.
                       Shape: [batch_size, truncated_backpropagation_length]
        :param processed_cnn_features: Output of the CNN from the previous module.
        :param initial_hidden_state: The initial hidden state of the RNN.
        :param output_layer: The final layer to be used to produce the output. Uses RNN's final output as input.
                             It is an instance of nn.Linear
        :param embedding_layer: The layer to be used to generate input embeddings for the RNN.
        :param is_train: Boolean variable indicating whether you're training the network or not.
                         If training mode is off then the predicted token should be fed as the input
                         for the next step in the sequence.

        :return: A tuple (logits, final hidden state of the RNN).
                 logits' shape = [batch_size, truncated_backpropagation_length, vocabulary_size]
                 hidden layer's shape = [num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train:
            sequence_length = tokens.shape[1]  # Truncated backpropagation length; 25 I think
        else:
            sequence_length = 40  # Max sequence length to generate

        # Get embeddings for the whole sequence
        embeddings = embedding_layer(input=tokens)  # Should have shape (batch_size, sequence_length, embedding_size)

        logits_sequence = []
        current_hidden_state = initial_hidden_state  #Initial hidden state shape: torch.Size([2, 128, 512])

        # TODO: Fetch the first (index 0) embeddings that should go as input to the RNN.
        # Use these tokens in the loop(s) below
        input_tokens = None  # Should have shape (batch_size, embedding_size)
        input_tokens = embeddings[:, 0, :]

        # Loop over the time steps from 0 to sequence_length:
        for i in range(0, sequence_length):
            # TODO:
            # 1. Loop over the RNN layers and provide them with correct input. Inputs depend on the layer
            #    index so input for layer-0 will be different from the input for other layers.
            # 2. Update the hidden cell state for every layer.
            # 3. If you are at the last layer, then produce logits_i, predictions. Append logits_i to logits_sequence.
            #    See the simplified rnn for the one layer version.

            updated_hidden_states = []  # list to store the hidden states of the 2 layers

            for j, cell in enumerate(self.cells):
                if j == 0:
                    # print(shape processed_cnn_features): torch.Size([128/8, 512])
                    # print(input_tokens.shape)): torch.Size([128/8, 300])
                    rnn_input = torch.cat((input_tokens, processed_cnn_features), dim=1)

                    # rnn_input = rnn_input.unsqueeze(0)  # to get [1, 128/8, 812]
                    # print("input_tokens.shape j=0: ", rnn_input.shape)
                else:
                    rnn_input = rnn_output.squeeze(0)
                # print("Input to RNN cell:", rnn_input.shape)

                # Here update weightes of the current hidden state and output of the cell
                # use unsqueeze(0) to pass the correct input dimensions.
                # print("shape current_hidden_state: ", current_hidden_state.shape)
                # print("shape current_hidden_state[j]: ", current_hidden_state[j].unsqueeze(0).shape)

                # new_hidden_state = cell(rnn_input, current_hidden_state[j].unsqueeze(0))
                new_hidden_state = cell(rnn_input, current_hidden_state[j])

                # remove (squeeze) first dim before appending to the list
                # make a list that contains the updated hidden state tensors for each layer in the RNN
                # Each hidden state tensor has a shape of [batch_size, hidden_state_size].

                #updated_hidden_states.append(hidden_state.squeeze(0))

                # For the next layer, use the rnn_output as input
                # rnn_input = rnn_output

                if isinstance(cell, LSTMCell):
                    hidden_state = new_hidden_state[:, :self.hidden_state_size]
                    cell_state = new_hidden_state[:, self.hidden_state_size:]

                    updated_hidden_states.append(torch.cat((hidden_state, cell_state), dim=1))
                    rnn_output = hidden_state.unsqueeze(0)
                else:
                    updated_hidden_states.append(new_hidden_state.squeeze(0))
                    rnn_output = new_hidden_state.unsqueeze(0)

            current_hidden_state = torch.stack(updated_hidden_states, dim=0)
            # updated_hidden_states list is stacked into a tensor along dimension 0
            # gives: [num_rnn_layers, batch_size, hidden_state_size] / [2, 128, 512]

            # output_layer() is an instance of the nn.Linear: calling the forward method of the nn.Linear class on the rnn_output tensor
            # logits_i = output_layer(current_hidden_state[0, :])
            logits_i = output_layer(current_hidden_state[-1])  # current_hidden_state[-1] -> 2D with last layer
            logits_sequence.append(logits_i)  # shape of logits_i -> [batch_size, vocabulary_size]

            # predictions, is a tensor with shape [batch_size],
            # where each element is the predicted index of the most probable word in the vocabulary
            predictions = torch.argmax(logits_i, dim=-1)
            # print("shape predictions", predictions.shape)

            # Get the input tokens for the next step in the sequence
            if i < sequence_length - 1:
                if is_train:
                    input_tokens = embeddings[:, i + 1, :]
                    # print("input_tokens shape time-point TRAIN/CORRECT:", input_tokens.shape): torch.Size([128,300])
                else:
                    # TODO: Compute predictions above and use them here by replacing None with the code in comment
                    # input_tokens = None  # embedding_layer(predictions)
                    input_tokens = embedding_layer(predictions)
                    # print("input_tokens shape time-point VALIDATE_2:", input_tokens.shape): torch.Size([8, 300])
                    # embedding_layer is an instance of nn.Embedding that maps word indices to their corresponding word embeddings
                    # input_tokens: each row represents the word embeddings of the predicted word for the corresponding input sample in the batch.


        logits = torch.stack(logits_sequence, dim=1)  # convert sequence of logits to a tensor

        # print("shape logits_i", logits_i.shape)  /  print("shape logits", logits.shape)
        # shape  logits_i: torch.Size([1, 128, 10000]) /  torch.Size([1, 8, 10000])
        # shape  logits: torch.Size([128, 25, 10000])  /  torch.Size([8, 25, 10000])

        return logits, current_hidden_state

########################################################################################################################


class GRUCell(nn.Module):
    def __init__(self, hidden_state_size: int, input_size: int):
        """
        :param hidden_state_size: Size (number of units/features) in the hidden state of GRU
        :param input_size: Size (number of units/features) of the input to the GRU
        """
        super(GRUCell, self).__init__()
        self.hidden_state_sizes = hidden_state_size

        # TODO: Initialise weights and biases for the update gate (weight_u, bias_u), reset gate (w_r, b_r) and hidden
        #       state (weight, bias).
        #       self.weight, self.weight_(u, r):
        #           A nn.Parameter with shape [HIDDEN_STATE_SIZE + input_size, HIDDEN_STATE_SIZE].
        #           Initialized using variance scaling with zero mean.
        #       self.bias, self.bias_(u, r): A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.
        #
        #       Tips:
        #           Variance scaling: Var[W] = 1/n

        # Update gate parameters
        self.weight_u = torch.nn.Parameter(
            torch.randn(hidden_state_size + input_size, hidden_state_size) / np.sqrt(hidden_state_size + input_size))
        self.bias_u = torch.nn.Parameter(torch.zeros(1, hidden_state_size))
        # Reset gate parameters
        self.weight_r = torch.nn.Parameter(
            torch.randn(hidden_state_size + input_size, hidden_state_size) / np.sqrt(hidden_state_size + input_size))
        self.bias_r = torch.nn.Parameter(torch.zeros(1, hidden_state_size))
        # Hidden state parameters
        self.weight = torch.nn.Parameter(
            torch.randn(hidden_state_size + input_size, hidden_state_size) / np.sqrt(hidden_state_size + input_size))
        self.bias = torch.nn.Parameter(torch.zeros(1, hidden_state_size))

    def forward(self, x, hidden_state):
        """
        Implements the forward pass for a GRU unit.
        :param x: A tensor with shape [batch_size, input_size] containing the input for the GRU.
        :param hidden_state: A tensor with shape [batch_size, HIDDEN_STATE_SIZE]
        :return: The updated hidden state of the GRU cell. Shape: [batch_size, HIDDEN_STATE_SIZE]
        """
        # TODO: Implement the GRU equations to get the new hidden state and return it
        # concatenation of the input x and the previous hidden state
        # print("IN GRU")
        # print(x.shape)
        # print(hidden_state.shape)
        # print(self.weight.shape)
        input_hidden = torch.cat((x, hidden_state), dim=1)

        # update gate
        u = torch.sigmoid(torch.matmul(input_hidden, self.weight_u) + self.bias_u)
        # reset gate
        r = torch.sigmoid(torch.matmul(input_hidden, self.weight_r) + self.bias_r)
        # proposed activation/candidate hidden state

        h_hat = torch.tanh(torch.matmul(torch.cat((r*hidden_state, x), dim=1), self.weight) + self.bias)

        # final output/new hidden state
        new_hidden_state = u * hidden_state + (1 - u) * h_hat

        return new_hidden_state

######################################################################################################################


class LSTMCell(nn.Module):
    def __init__(self, hidden_state_size: int, input_size: int):
        """
        :param hidden_state_size: Size (number of units/features) in the hidden state of GRU
        :param input_size: Size (number of units/features) of the input to GRU
        """
        super(LSTMCell, self).__init__()
        self.hidden_state_size = hidden_state_size

        # TODO: Initialise weights and biases for the forget gate (weight_f, bias_f), input gate (w_i, b_i),
        #       output gate (w_o, b_o), and hidden state (weight, bias)
        #       self.weight, self.weight_(f, i, o):
        #           A nn.Parameter with shape [HIDDEN_STATE_SIZE + input_size, HIDDEN_STATE_SIZE].
        #           Initialized using variance scaling with zero mean.
        #       self.bias, self.bias_(f, i, o): A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to two.
        #
        #       Tips:
        #           Variance scaling: Var[W] = 1/n
        #       Note: The actual input tensor will have 2 * HIDDEN_STATE_SIZE because it contains both
        #             hidden state and cell's memory
        #       Note: the input hidden state should be (num_layers, batch_size, hidden_dim * 2),
        #       as the first half represents the hidden state and the second half represents the memory cell state

        # Initialise weights and biases:
        # Forget gate parameters
        self.weight_f = torch.nn.Parameter(
            torch.randn(hidden_state_size + input_size, hidden_state_size) / np.sqrt(hidden_state_size + input_size))
        self.bias_f = torch.nn.Parameter(torch.zeros(1, hidden_state_size))
        # Input gate parameters
        self.weight_i = torch.nn.Parameter(
            torch.randn(hidden_state_size + input_size, hidden_state_size) / np.sqrt(hidden_state_size + input_size))
        self.bias_i = torch.nn.Parameter(torch.zeros(1, hidden_state_size))
        # Output gate parameters
        self.weight_o = torch.nn.Parameter(
            torch.randn(hidden_state_size + input_size, hidden_state_size) / np.sqrt(hidden_state_size + input_size))
        self.bias_o = torch.nn.Parameter(torch.zeros(1, hidden_state_size))
        # Memory cell parameters
        self.weight = torch.nn.Parameter(
            torch.randn(hidden_state_size + input_size, hidden_state_size) / np.sqrt(hidden_state_size + input_size))
        self.bias = torch.nn.Parameter(torch.zeros(1, hidden_state_size))


    def forward(self, x, hidden_state):
        """
        Implements the forward pass for an LSTM unit.
        :param x: A tensor with shape [batch_size, input_size] containing the input for the GRU.
        :param hidden_state: A tensor with shape [batch_size, 2 * HIDDEN_STATE_SIZE] containing the hidden
                             state and the cell memory. The 1st half represents the hidden state and the
                             2nd half represents the cell's memory
        :return: The updated hidden state (including memory) of the GRU cell.
                 Shape: [batch_size, 2 * HIDDEN_STATE_SIZE]
        """
        # TODO: Implement the LSTM equations to get the new hidden state, cell memory and return them.
        #       The first half of the returned value must represent the new hidden state and the second half
        #       new cell state.

        # splits hidden_state tensor: 1) previous hidden state h_prev; 2) previous memory cell state c_pre
        h_prev = hidden_state[:, :self.hidden_state_size]
        c_prev = hidden_state[:, self.hidden_state_size:]

        # concatenation of the input x and the previous hidden state
        input_hidden = torch.cat((x, h_prev), dim=1)  # shape (batch_size, input_size + hidden_state_size)

        # compute gates:
        i_t = torch.sigmoid(torch.matmul(input_hidden, self.weight_i) + self.bias_i)
        o_t = torch.sigmoid(torch.matmul(input_hidden, self.weight_o) + self.bias_o)
        f_t = torch.sigmoid(torch.matmul(input_hidden, self.weight_f) + self.bias_f)
        c_hat_t = torch.tanh(torch.matmul(input_hidden, self.weight) + self.bias)

        # print("Shape of c_prev:", c_prev.shape)

        # computes new memory cell state c_t
        c_t = f_t * c_prev + i_t * c_hat_t
        # computes new hidden state h_t
        h_t = o_t * torch.tanh(c_t)

        new_hidden_state = torch.cat((h_t, c_t), dim=1)
        return new_hidden_state

######################################################################################################################


def loss_fn(logits, y_tokens, y_weights):
    """
    Weighted softmax cross entropy loss.

    Args:
        logits           : Shape[batch_size, truncated_backprop_length, vocabulary_size]
        y_tokens (labels): Shape[batch_size, truncated_backprop_length]
        y_weights         : Shape[batch_size, truncated_backprop_length]. Add contribution to the total loss only
                           from words existing
                           (the sequence lengths may not add up to #*truncated_backprop_length)

    Returns:
        sum_loss: The total cross entropy loss for all words
        mean_loss: The averaged cross entropy loss for all words

    Tips:
        F.cross_entropy
    """
    eps = 1e-7  # Used to avoid division by 0

    logits = logits.view(-1, logits.shape[2])
    y_tokens = y_tokens.view(-1)
    y_weights = y_weights.view(-1)
    losses = F.cross_entropy(input=logits, target=y_tokens, reduction='none')

    sum_loss = (losses * y_weights).sum()
    mean_loss = sum_loss / (y_weights.sum() + eps)

    return sum_loss, mean_loss

# #####################################################################################################################
# if __name__ == '__main__':
#
#     lossDict = {'logits': logits,
#                 'yTokens': yTokens,
#                 'yWeights': yWeights,
#                 'sumLoss': sumLoss,
#                 'meanLoss': meanLoss
#     }
#
#     sumLoss, meanLoss = loss_fn(logits, yTokens, yWeights)
#

######################################################################################################################
