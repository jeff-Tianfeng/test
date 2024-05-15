from GRU import GRU
import torch
import numpy as np
import torch.nn as nn
import time
import sys


def generate_copy_sequence(length, size, batch_size):
    np.random.seed(42)
    sequences = []
    for batch in range(batch_size):
        sequence = np.random.randint(2, size=(length, size))  # Random binary sequence
        sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(1)
        sequences.append(sequence)
    # sequences = torch.stack(sequences, dim = 0)
    # sequences = sequences.squeeze(1).squeeze(1)
    return sequences, sequences  # Input sequence and target sequence are the same


def validate_model(model, seq_length, input_size, output_size):
    model.eval()
    loss = 0
    # input_seq, target_seq = generate_copy_sequence(seq_length, input_size)
    # input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(1)
    # target_tensor = torch.tensor(target_seq, dtype=torch.float32).unsqueeze(1)
    input_tensor = torch.tensor([[[0., 1., 0., 0.]]])
    target_tensor = torch.tensor([[[0., 1., 0., 0.]]])
    hidden = torch.zeros(1, 1, hidden_size)  # Initial hidden state

    with torch.no_grad():
        for epoch in range(1000):
            output, _, grad = model(input_tensor, hidden)
            predicted_sequence = torch.round(output.squeeze().detach()).numpy()  # Round to binary values
            loss += criterion(output, target_tensor)

        print("Loss in validation is:", loss.item() / 1000)
        print("Input sequence:")
        print(input_tensor)
        print("Target sequence:")
        print(target_tensor)
        print("Predicted sequence:")
        print(output)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    last_update_percent = 0

    # 设置刷新频率阈值
    refresh_threshold = 1
    # how many columns the test data has
    input_size = 1
    # turn this param to set the size of model (model params counts)
    # 32 would perform well on this task, with 4554 params
    # 16 is a bit poorer than 32, which holds 1514 params
    hidden_size = 128
    # make sure the output_size is identical to the input_size, because this is a copy task
    output_size = 1
    # how many rows the test data has
    seq_length = 1

    num_epochs = 1000
    batch_size = 1

    learning_rate = 0.1
    learning_rate= learning_rate
    # if sparsity = 0 then dense network
    # else sparse network
    sparsity = 0
    snapLevel = 2
    rank = 2
    # snap_Level = 0
    type = 'SnAp'

    # Initialize model, loss function
    model = GRU(batch_size, 1, hidden_size, 1, sparsity, snapLevel, seq_length, type, rank ,device).to(device)
    # print the model parameter size
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model built, total trainable params: " + str(total_params))
    # loss criterion
    criterion = nn.MSELoss()
    # if needed, use the optimizer to update params
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss = 0
    output = 0
    # storage for BPTT
    # inputs = []
    hiddens = []
    # Dts = []
    # Dh_Dthetas = []
    # data for draw graph
    data_times = [0]
    sequence_length = [1]
    # Training loop
    start_time = time.time()
    jt = torch.zeros((model.hidden_size, model.params.size()[0] * model.params.size()[1])).to(device)
    for epoch in range(num_epochs):
        batchloss = 0
        bit_losses = 0
        hiddens = []
        input_seq, target_seq = generate_copy_sequence(seq_length, input_size, batch_size)
        hidden = torch.zeros(hidden_size, requires_grad=True).to(device)
        for rang in range(len(input_seq[0][0][0])):
            input_tensor = input_seq[0].to(device)
            target_tensor = target_seq[0].to(device)
            # BPTT
            if type == 'BPTT':
                output, hidden = model(input_tensor[0][0][rang], hidden)
                loss = model.bptt(output, target_tensor[0][0][rang])
                # hiddens.append(hidden)
            # RTRL
            if type == 'RTRL':
                old_hidden = hidden
                output, hidden = model(input_tensor[0][0][rang], hidden)
                loss, jt = model.rtrl(output, target_tensor[0][0][rang], hidden, old_hidden, jt, learning_rate)
            # SnAp
            if type == 'SnAp':
                old_hidden = hidden
                output, hidden = model(input_tensor[0][0][rang], hidden)
                loss, jt = model.snap1(output, target_tensor[0][0][rang], hidden, old_hidden, jt, learning_rate)


            with torch.no_grad():
                batchloss += loss
                bit_losses += torch.abs(output - target_tensor[0][0][rang])

        if batchloss / batch_size < 0.1:
            learning_rate = 0.01
        if batchloss / batch_size < 0.01:
            learning_rate = 0.001

        all_losses_below_threshold = torch.div(bit_losses, len(input_seq[0][0][0])) < 0.15
        if all_losses_below_threshold:
            input_size += 1
            learning_rate = 0.1
            data_times.append(epoch * batch_size)
            sequence_length.append(input_size)
            output_size = input_size
            model = GRU(batch_size, 1, hidden_size, 1, sparsity, snapLevel, seq_length, type, rank, device).to(device)
            input_seq, target_seq = generate_copy_sequence(seq_length, input_size, batch_size)
            jt = torch.zeros((model.hidden_size, model.params.size()[0] * model.params.size()[1])).to(device)
            hidden = torch.zeros(hidden_size, requires_grad=True).to(device)
            loss = 0
            # clear memory of BPTT
            # inputs = []
            # hiddens = []
            # Dts = []
            # Dh_Dthetas = []
        progress_percent = (epoch + 1) / num_epochs * 100

        # 计算当前进度与上次更新进度的差值
        progress_diff = progress_percent - last_update_percent

        # 如果差值大于等于刷新频率阈值，就更新进度
        sys.stdout.write(
            f'\rEpoch [{epoch + 1}/{num_epochs}], Progress: {progress_percent:.2f}%, loss: {batchloss / batch_size}, {sequence_length}')
        # , sequence: {torch.div(bit_losses, batch_size)[0][0]}
        sys.stdout.flush()
    end_time = time.time()
    print("data times for BPTT:")
    print(data_times)
    print("sequences lengths for BPTT")
    print(sequence_length)
    print("time cost")
    print(end_time - start_time)



        # show progress
    #     bit_losses += torch.abs(output - target_tensor)
    #     print(output)
    #     print(target_tensor)
    #     progress_percent = epoch / num_epochs * 100
    #     sys.stdout.write(f'\rEpoch [{epoch + 1}/{num_epochs}], Progress: {progress_percent:.2f}%')
    #     sys.stdout.flush()
    #     # bit loss judge
    #     if epoch % batch_size == 0:
    #         all_losses_below_threshold = torch.all(torch.div(bit_losses, batch_size) < 0.15)
    #         if all_losses_below_threshold:
    #             print(input_size)
    #             input_size += 1
    #             output_size = input_size
    #             model = GRU(input_size, hidden_size, output_size, sparsity, snapLevel, seq_length, type, rank)
    #             jt = torch.zeros_like(model.params)
    #             input_seq, target_seq = generate_copy_sequence(seq_length, input_size)
    #             input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(1)
    #             target_tensor = torch.tensor(target_seq, dtype=torch.float32).unsqueeze(1)
    #             hidden = torch.zeros(1, 1, hidden_size)
    #             # clear memory of BPTT
    #             inputs = []
    #             hiddens = []
    #             Dts = []
    #             Dh_Dthetas = []
    #         # Clean loss calculator
    #         bit_losses = 0
    #     if type == 'BPTT':
    #         losses_BPTT.append(input_size)
    #     elif type == 'SnAp' and snapLevel == 1:
    #         losses_SnAp_1.append(input_size)
    #     elif type == 'SnAp' and snapLevel == 2:
    #         losses_SnAp_2.append(input_size)
    #     elif type == 'SnAp' and snapLevel == 3:
    #         losses_SnAp_3.append(input_size)
    #     elif type == 'RTRL':
    #         losses_RTRL.append(input_size)
    #     # validate the model using human_readable outputs
    # end_time = time.time()
    # plt.plot(losses_BPTT, label='BPTT')
    # plt.plot(losses_SnAp_1, label='SnAp_1')
    # plt.plot(losses_SnAp_2, label='SnAp_2')
    # plt.plot(losses_SnAp_3, label='SnAp_3')
    # plt.plot(losses_RTRL, label='RTRL')
    # plt.xlabel('data time')
    # plt.ylabel('Sequence Length')
    # plt.title('performance')
    # plt.legend()
    # plt.show()
    # validate_model(model, seq_length, input_size, output_size)
    # training_time = end_time - start_time
    # print("Training time: {:.2f} seconds".format(training_time))
