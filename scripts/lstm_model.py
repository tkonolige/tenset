import tvm

from tir_dataset import TIRPrograms
import sqlite3_dataset

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.autograd.profiler as profiler

import numpy as np
import cProfile


device = torch.device("cuda")


class TokenEmbedding(nn.Module):
    def __init__(self):
        super(TokenEmbedding, self).__init__()
        self.embedding_dim = 179
        self.builtin_embedding = nn.Embedding(122, self.embedding_dim, device=device)
        self.variable_embedding = nn.Embedding(
            10000, self.embedding_dim, device=device
        )  # max 100 variables?

    def forward(self, tokens):
        out = torch.zeros(tokens.shape[0], self.embedding_dim + 1, device=device)
        is_builtin = tokens[:, 1] == 0
        out[is_builtin, : self.embedding_dim] = self.builtin_embedding(tokens[is_builtin, 0])
        out[is_builtin, self.embedding_dim] = -1
        is_variable = tokens[:, 1] == 1
        out[is_variable, : self.embedding_dim] = self.variable_embedding(tokens[is_variable, 0])
        out[is_variable, self.embedding_dim] = -1
        is_number = tokens[:, 1] == 1
        out[is_number, : self.embedding_dim] = 0
        out[is_number, self.embedding_dim] = tokens[is_number, 0].float()
        return out


class LSTMPredictor(nn.Module):
    def __init__(self):
        super(LSTMPredictor, self).__init__()
        self.token_embedding = TokenEmbedding()
        self.hidden_stmt_size = 180
        self.hidden_loop_size = 180
        self.stmt_reducer = nn.LSTM(
            self.token_embedding.embedding_dim + 1, self.hidden_stmt_size, device=device
        )
        self.stmt_ff = nn.Sequential(nn.Linear(180, 180), nn.ELU())
        self.loop_reducer = nn.LSTM(
            self.token_embedding.embedding_dim + 1, self.hidden_loop_size, device=device
        )
        self.loop_ff = nn.Sequential(nn.Linear(180, 180), nn.ELU())
        self.tokenize_fn = tvm.get_global_func("tir.analysis.tokenize_stmt")()
        self.ff_final = nn.Sequential(
            nn.Linear(180, 90),
            nn.ELU(),
            nn.Linear(90, 1),
        )

    def tokenize(self, ast):
        return self.tokenize_fn(ast).asnumpy()

    def tokenize_for(self, f):
        # TODO: missing for kind
        tokens = np.concatenate(
            [self.tokenize(f.min), self.tokenize(f.extent), self.tokenize(f.loop_var)]
        )
        return self.token_embedding(torch.as_tensor(tokens, device=next(self.parameters()).device))

    def visit(self, ast):
        if isinstance(ast, tvm.tir.stmt.For):
            children = self.visit(ast.body)
            embedded = self.tokenize_for(ast)
            y, hidden = self.loop_reducer(embedded.view(embedded.shape[0], 1, embedded.shape[1]))
            y = hidden[0]
            for x in children:
                y, hidden = self.loop_reducer(x.view(x.shape[0], 1, -1), hidden)
            x = self.loop_ff(hidden[0])
            return x

        if isinstance(ast, tvm.tir.stmt.Allocate):
            # TODO: include this information?
            return self.visit(ast.body)

        if isinstance(ast, tvm.tir.stmt.AttrStmt):
            # TODO: include this information?
            return self.visit(ast.body)

        if isinstance(ast, tvm.tir.stmt.SeqStmt):
            return [self.visit(x) for x in ast.seq]

        embedded = self.token_embedding(
            torch.as_tensor(self.tokenize(ast), device=next(self.parameters()).device)
        )
        out, hidden = self.stmt_reducer(embedded.view(embedded.shape[0], 1, embedded.shape[1]))
        x = self.stmt_ff(out)
        return x

    def forward(self, ast):
        x = self.visit(ast.body)
        # TODO: figure out a better way to handle this
        if isinstance(x, list):
            y = None
            hidden = (
                torch.zeros(1, 1, self.hidden_loop_size, device=device),
                torch.zeros(1, 1, self.hidden_loop_size, device=device),
            )
            for z in x:
                y, hidden = self.loop_reducer(z, hidden)
            return self.ff_final(y.reshape(-1, self.hidden_loop_size))
        return self.ff_final(x.reshape(-1, self.hidden_loop_size))


def train(model):
    dataset = sqlite3_dataset.TIRPrograms(
        "dataset_cpu.db", "dataset_cpu/network_info/all_tasks.pkl"
    )
    loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    batch_size = 16

    for epoch in range(50):
        for i, ((target, tir), cost) in enumerate(dataset.iterate()):
            if i % batch_size == 0:
                optimizer.zero_grad()

            predicted_cost = model(tir)
            d = loss(predicted_cost, torch.tensor([[cost]], dtype=torch.float32, device=device))
            print(f"{i:10}{predicted_cost[0,0,0]:10.2e}{cost:10.2e}{d.item():10.2e}")
            d.backward()
            if i % batch_size == batch_size - 1:
                optimizer.step()

            # if i % 100 == 0:
            #     with torch.no_grad():
            #         (target, tir), cost = dataset[0]
            #         predicted_cost = model(tir)
            #         print(predicted_cost, cost)


class SingleLayerLSTM(nn.Module):
    def __init__(self):
        super(SingleLayerLSTM, self).__init__()
        self.embedding = TokenEmbedding()
        self.lstm = nn.LSTM(180, 180)
        self.ff = nn.Sequential(
            nn.Linear(180, 80),
            nn.ELU(),
            nn.Linear(80, 1),
        )
        self.tokenize_fn = tvm.get_global_func("tir.analysis.tokenize_stmt")()

    def forward(self, ast):
        tks = self.tokenize_fn(ast.body).asnumpy()
        embedded = self.embedding(torch.as_tensor(tks, device=device))
        out, hidden = self.lstm(embedded.view(-1, 1, embedded.shape[1]))
        return self.ff(hidden[0])


if __name__ == "__main__":
    # progs = TIRPrograms("dataset_cpu/network_info/all_tasks.pkl", "dataset_cpu/measure_records")
    # (target, tir), cost = progs[0]
    # import pickle
    # tir = pickle.load(open("tir_program.pkl", "rb"))
    # model = LSTMPredictor()
    # print(model(tir))
    # model = LSTMPredictor().cuda()
    model = SingleLayerLSTM().cuda()
    train(model)
