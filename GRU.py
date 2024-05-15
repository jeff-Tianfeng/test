import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim


def low_rank_approximation(tensor, rank):
    U, S, V = torch.svd(tensor)
    k = rank
    # using the top k eigenvalues to recompose low rank matrix
    # S_k = torch.diag(S[:k])
    U_k = U[:, :k]
    V_k = V[:, :k]
    p = torch.matmul(torch.matmul(U_k, torch.matmul(torch.matmul(torch.transpose(U_k, 0, 1), tensor), V_k)),
                     torch.transpose(V_k, 0, 1))
    # tensor_approx = torch.mm(torch.mm(U_k, S_k), V_k.t())
    return p


class GRU(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, output_size, sparsity, snapLevel
                 , seq_length, type, rank, device):
        super(GRU, self).__init__()
        torch.manual_seed(40)
        self.input_size = input_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.criterion = nn.MSELoss()
        self.sparsity = torch.tensor(sparsity)
        self.rank = rank
        self.type = type
        self.device = device

        # Reset gate params
        self.W_ir = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.Tensor(1, hidden_size))
        # Update gate params
        self.W_iz = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(1, hidden_size))
        # New memory gate params
        self.W_in = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_n = nn.Parameter(torch.Tensor(1, hidden_size))
        # output layer params
        self.fc = nn.Linear(hidden_size, output_size)
        self.init_weights()
        self.params = torch.cat((
            self.W_ir, self.W_hr, self.b_r,
            self.W_iz, self.W_hz, self.b_z,
            self.W_in, self.W_hn, self.b_n))
        self.parammask = []
        self.hiddenMaks = None
        self.snapLevel = snapLevel
        self.apply_sparsity()
        self.W_ir_index = np.array([i for i in range(input_size)])
        self.W_hr_index = np.array([i + input_size for i in range(hidden_size)])
        self.b_r_index = np.array([input_size + hidden_size])
        self.W_iz_index = np.array([i + input_size + hidden_size + 1 for i in range(input_size)])
        self.W_hz_index = np.array([i + (2 * input_size) + hidden_size + 1 for i in range(hidden_size)])
        self.b_z_index = np.array([2 * (input_size + hidden_size) + 1])
        self.W_in_index = np.array([i + 2 * (input_size + hidden_size) + 2 for i in range(input_size)])
        self.W_hn_index = np.array([i + 2 * (input_size + hidden_size) + input_size + 2 for i in range(hidden_size)])
        self.b_n_index = np.array([3 * (input_size + hidden_size) + 2])
        if type == "SnAp":
            self.cal_Snap()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.nonzero_indices = None

    def init_weights(self):
        nn.init.xavier_uniform_(self.W_ir)
        nn.init.xavier_uniform_(self.W_hr)
        nn.init.constant_(self.b_r, 1)

        nn.init.xavier_uniform_(self.W_iz)
        nn.init.xavier_uniform_(self.W_hz)
        nn.init.constant_(self.b_z, 1)

        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_hn)
        nn.init.constant_(self.b_n, 1)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 1)

    def apply_sparsity(self):
        if self.type == 'BPTT':
            with torch.no_grad():
                for param in self.parameters():
                    if len(param.size()) > 1:
                        mask = torch.rand_like(param) > self.sparsity
                        param.mul_(mask.float())
                        self.parammask.append(mask)
        else:
            with torch.no_grad():
                mask = torch.rand_like(self.params) > self.sparsity
                self.params.mul_(mask.float())
                self.parammask = self.params == 0

    def forward(self, input, hidden):
        # Reset gate
        # normed_input = input.clone().detach().float()
        normed_input = torch.unsqueeze(input, 0)
        normed_input = torch.unsqueeze(normed_input, 0).to(self.device)
        hidden = hidden.to(self.device)

        r = torch.sigmoid(torch.matmul(normed_input, self.params[[self.W_ir_index]].to(self.device)) +
                          torch.matmul(hidden, self.params[[self.W_hr_index]].to(self.device)) + self.params[[self.b_r_index]].to(self.device))
        # Update gate
        z = torch.sigmoid(torch.matmul(normed_input, self.params[[self.W_iz_index]].to(self.device)) +
                          torch.matmul(hidden, self.params[[self.W_hz_index]].to(self.device)) + self.params[[self.b_z_index]].to(self.device))
        # New memory gate
        n = torch.tanh(torch.matmul(normed_input, self.params[[self.W_in_index]].to(self.device)) +
                       r * (torch.matmul(hidden, self.params[[self.W_hn_index]].to(self.device)) + self.params[[self.b_n_index]].to(self.device)))
        # Hidden state
        hidden_new = (1 - z) * hidden + z * n
        output = self.fc(hidden_new)
        return output, hidden_new

    # def snap1(self, output, targets, hidden, old_hidden, jt, learning_rate):
    #     # initialize the Jt matrix
    #     updated_jt = jt
    #     loss = self.criterion(output[0][0], targets[0][0])
    #     d_loss_d_hidden = torch.autograd.grad(loss, hidden, retain_graph=True)[0]
    #     p = torch.autograd.grad(hidden, old_hidden, grad_outputs=d_loss_d_hidden, retain_graph=True)[0]
    #     Dh_Dtheta = torch.autograd.grad(hidden, self.params, grad_outputs=d_loss_d_hidden, retain_graph=True)[0]
    #     # if self.nonzero_indices is None:
    #     self.nonzero_indices = torch.nonzero(torch.sum(torch.abs(Dh_Dtheta), dim=1))
    #     # apply the snap mask
    #     jt[self.hiddenMaks] = 0
    #     # Traverse the input sequence and calculate the gradient
    #     Dt_Jt = torch.matmul(jt[self.nonzero_indices], p)
    #     updated_jt[self.nonzero_indices] += Dh_Dtheta[self.nonzero_indices] + Dt_Jt
    #     with torch.no_grad():
    #         self.params[self.nonzero_indices] -= learning_rate * updated_jt[self.nonzero_indices]
    #         self.params[self.parammask] = 0
    #     return loss, updated_jt

    def snap1(self, output, targets, hidden, old_hidden, jt, learning_rate):
        # initialize the Jt matrix
        # updated_jt = torch.zeros((self.hidden_size, self.params.size()[0] * self.params.size()[1]))
        loss = self.criterion(output, targets)

        d_loss_d_hidden = torch.autograd.grad(loss, hidden, retain_graph=True)[0]
        Dt = torch.zeros((self.hidden_size, self.hidden_size))
        It = torch.zeros((self.hidden_size, self.params.size()[0], self.params.size()[1]))
        updated_jt = torch.zeros((self.hidden_size, self.params.size()[0] * self.params.size()[1]))
        for i in range(self.hidden_size):
            Dt[i, :] = torch.autograd.grad(hidden[0][i], old_hidden, retain_graph=True)[0]
            It[i, :, :] = torch.autograd.grad(hidden[0][i], self.params, retain_graph=True)[0]
        It = It.view(self.hidden_size, -1)
        jt[self.hiddenMaks] = 0
        nonzero_indices = torch.nonzero(jt)
        nonzero_columns = nonzero_indices[:, 1]
        start_time = time.time()
        if nonzero_columns.numel() == 0:
            Dt_Jt = torch.matmul(Dt, jt)
            updated_jt = It + Dt_Jt
        else:
            Dt_Jt = torch.matmul(Dt, jt[:, nonzero_columns])
            updated_jt[:, nonzero_columns] = It[:, nonzero_columns] + Dt_Jt
        end_time = time.time()
        p = torch.matmul(d_loss_d_hidden[0], updated_jt)
        gd = p.view(self.params.size()[0], self.params.size()[1])
        # print(end_time - start_time)
        with torch.no_grad():
            self.params -= learning_rate * gd
            self.params[self.parammask] = 0
        return loss, updated_jt

    # def compute_sparsity_pattern(self, input_sequence):
    #     sparsity_pattern = []
    #     for input_step in input_sequence:
    #         _, hidden_state = self(input_step)
    #         hidden_state.retain_grad()
    #         hidden_state.backward(torch.ones_like(hidden_state), retain_graph=True)
    #         sparsity_pattern.append((hidden_state.grad != 0).detach())
    #     return sparsity_pattern
    def cal_Snap(self):

        def generate_copy_sequence(length, size):
            sequence = np.random.randint(2, size=(length, size))  # Random binary sequence
            return sequence, sequence  # Input sequence and target sequence are the same

        input_seq, target_seq = generate_copy_sequence(self.seq_length, self.input_size)
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(1)
        hidden = torch.zeros(self.hidden_size, requires_grad=True)
        It = torch.zeros((self.hidden_size, self.params.size()[0], self.params.size()[1]))
        for i in range(self.snapLevel + 1):
            output, hidden = self.forward(input_tensor[0][0][0], hidden)
        for j in range(self.hidden_size):
            It[j, :, :] = torch.autograd.grad(hidden[0][j], self.params, retain_graph=True)[0]
        with torch.no_grad():
            It = It.view(self.hidden_size, -1)
            self.hiddenMaks = It == 0
            num_true = self.hiddenMaks.sum().item()
            # 计算张量的大小
            total_elements = self.hiddenMaks.numel()

            # 计算 False 的数量
            num_false = total_elements - num_true

            print("Number of True:", num_true)
            print("Number of False:", num_false)

    def bptt(self, output, targets):
        loss = self.criterion(output, targets)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
        self.optimizer.step()
        # Update weights and biases
        with torch.no_grad():
            for param, mask in zip(self.parameters(), self.parammask):
                param.mul_(mask.float())
        return loss

    # def bptt(self, output, inputs, targets, hidden, Dts, Dh_Dthetas, learning_rate):
    #     loss = self.criterion(output[0][0], targets[0][0])
    #     loss.backward(retain_graph=True)
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
    #     # Update weights and biases
    #     self.optimizer.step()
    #     self.params[self.parammask] = 0
    # d_loss_d_hidden = torch.autograd.grad(loss, hidden, retain_graph=True)[0]
    # Dh_Dtheta = torch.autograd.grad(hidden, self.params, grad_outputs=d_loss_d_hidden, retain_graph=True)[0]
    # Dh_Dthetas.append(Dh_Dtheta)
    # # low_rank_approximation(Dh_Dthetas[-1], self.rank)
    # if len(Dh_Dthetas) != 1:
    #     Dh_Dthetas[-1] = Dh_Dthetas[-1] + torch.matmul(Dh_Dthetas[-2], Dts[-1].squeeze())
    # with torch.no_grad():
    #     self.params -= learning_rate * Dh_Dthetas[-1]
    #     self.params[self.parammask] = 0
    # return loss, Dh_Dthetas

    # def init_hidden(self, batch_size):
    #     return torch.zeros(batch_size, self.hidden_size)
    def rtrl(self, output, targets, hidden, old_hidden, jt, learning_rate):

        device = self.params.device
        loss = self.criterion(output[0][0], targets)

        d_loss_d_hidden = torch.autograd.grad(loss, hidden, retain_graph=False)[0]
        Dt = torch.zeros((self.hidden_size, self.hidden_size)).to(device)
        It = torch.zeros((self.hidden_size, self.params.size()[0], self.params.size()[1])).to(device)
        # print(torch.autograd.grad(hidden[0], self.params, grad_outputs=torch.ones_like(hidden[0]), retain_graph=True)[0])
        for i in range(self.hidden_size):
            Dt[i, :] = torch.autograd.grad(hidden[0][i], old_hidden, retain_graph=True)[0]
            It[i, :, :] = torch.autograd.grad(hidden[0][i], self.params, retain_graph=True)[0]
        # print(It)
        It = It.view(self.hidden_size, -1)
        Dt_Jt = torch.matmul(Dt, jt)
        updated_jt = It + Dt_Jt

        p = torch.matmul(d_loss_d_hidden[0], updated_jt)
        gd = p.view(self.params.size()[0], self.params.size()[1])
        # print(e-s)
        with torch.no_grad():
            self.params -= learning_rate * gd
            self.params[self.parammask] = 0
        return loss, updated_jt

    # def rtrl(self, output, targets, hidden, old_hidden, jt, learning_rate):
    #     # initialize the Jt matrix
    #     updated_jt = jt
    #     loss = self.criterion(output, targets)
    #     d_loss_d_hidden = torch.autograd.grad(loss, hidden, retain_graph=True)[0]
    #     p = torch.autograd.grad(hidden, old_hidden, grad_outputs=d_loss_d_hidden, retain_graph=True)[0]
    #     Dh_Dtheta = torch.autograd.grad(hidden, self.params, grad_outputs=d_loss_d_hidden, retain_graph=True)[0]
    #     # Traverse the input sequence and calculate the gradient
    #     Dt_Jt = torch.matmul(jt, p)
    #     temp = Dh_Dtheta + Dt_Jt
    #     temp = torch.squeeze(temp)
    #     updated_jt += temp
    #     # updated_jt = low_rank_approximation(updated_jt, self.rank)
    #     # tensor_approx = low_rank_approximation(updated_jt, self.rank)
    #     with torch.no_grad():
    #         self.params -= learning_rate * updated_jt
    #         self.params[self.parammask] = 0
    #     return loss, updated_jt
