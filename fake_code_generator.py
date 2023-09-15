import numpy as np
import pandas as pd
import random
import polars as pl
import string
from torch.utils.data import DataLoader
import uuid
import os

df = pl.read_parquet('datafiles/data_with_code655.parquet')
df_chars=pl.read_parquet('opcode_to_char.parquet')
charlist=df_chars['characters'].to_list()
characters="".join(charlist)

def generateCode(df,characters):
    lengths = df['merged_opcodes'].str.len()
    values, counts = np.unique(lengths, return_counts=True)
    random_length = np.random.choice(values, p=counts/counts.sum())
    random_string = ''.join(random.choices(characters, k=int(random_length)))
    return random_string


import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch

torch.device("mps")

def tryLoad():
    files=os.listdir("./models/")
    for file in files:
        first=file[0]
        if first=="D":
            d=file
        else:
            g=file
    return d,g

class Generator(nn.Module):
    def __init__(self, input_length, output_length):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_length, output_length),
            nn.Conv1d(output_length, 128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_length):
        super(Discriminator, self).__init__()
        print(f"Expected input length for Discriminator: {input_length}")
        self.model = nn.Sequential(
            nn.Linear(input_length, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train(generator, discriminator, data_loader, epochs=10000):
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(),lr=0.0005)
    optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=0.0001)
    d,g=tryLoad()

    try:
        generator.load_state_dict(torch.load(g))
    except Exception:
        pass

    try:
        discriminator.load_state_dict(torch.load(d))
    except Exception:
        pass

    d_loss_min=1000
    g_loss_min=1000
    sum_min=1000
    for epoch in range(epochs):

        choice=random.choice([True,False,False,False,False,False,False,False,False,False])
        choice2=random.choice([True,False,False,False,False,False,False,False,False,False])

        if choice and epoch>1:
            generator.load_state_dict(torch.load(best_g))
        
        if choice2 and epoch>1:
            discriminator.load_state_dict(torch.load(best_d))

        for i, data in enumerate(data_loader):
            
            real_data = data.float()
            print(f"Shape of real_data: {real_data.shape}") 
            fake_data = generator(torch.randn(real_data.size(0), 100))
            fake_output = discriminator(fake_data.detach())
            fake_output = torch.transpose(fake_output, 0, 1)
            real_output = discriminator(real_data)
            d_loss = criterion(real_output, torch.ones(real_data.size(0), 1)) + criterion(fake_output, torch.zeros(real_data.size(0), 1))
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, torch.ones(real_data.size(0), 1))
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
        d_loss_curr=d_loss.item()
        g_loss_curr=g_loss.item()
        sum_curr=g_loss_curr+d_loss_curr
        if d_loss_curr<=d_loss_min or g_loss_curr<=g_loss_min or sum_curr<=sum_min:

            hashit=uuid.uuid1()
            if sum_curr<=sum_min:
                sum_min=sum_curr

            if d_loss_curr<=d_loss_min or sum_curr<=sum_min:
                d_loss_min=d_loss_curr
                try:
                    os.remove(best_d)
                except Exception:
                    pass
                best_d='models/D_{scribble}_{lossx}.pkl'.format(scribble=hashit,lossx=d_loss_min)
                torch.save(discriminator.state_dict(), best_d)
            
            if g_loss_curr<=g_loss_min or sum_curr<=sum_min:
                g_loss_min=g_loss_curr
                try:
                    os.remove(best_g)
                except Exception:
                    pass
                best_g='models/G_{scribble}_{lossx}.pkl'.format(scribble=hashit,lossx=g_loss_min)
                torch.save(generator.state_dict(), best_g)

        print(f"Epoch {epoch}/{epochs} D loss: {d_loss_curr} G loss: {g_loss_curr}")
        print(f"Best D: {best_d} Best G: {best_g}")

def prep_and_train():
    human_generated_strings = df['merged_opcodes'].apply(lambda x: [ord(c) for c in x]).tolist()
    human_generated_strings = [torch.tensor(lst) for lst in human_generated_strings]
    human_generated_strings = pad_sequence(human_generated_strings, batch_first=True)
    batch_size = 100
    data_loader = DataLoader(human_generated_strings, batch_size=batch_size, shuffle=True)
    generator = Generator(100, len(human_generated_strings[0]))
    discriminator = Discriminator(len(human_generated_strings[0]))
    train(generator, discriminator, data_loader)
    torch.save(generator.state_dict(), 'generator.pkl')
    torch.save(discriminator.state_dict(), 'discriminator.pkl')

#prep_and_train()

def generate_fake_string(generator, input_length, characters):
    noise = torch.randn(100, input_length)
    generated_output = generator(noise)
    generated_output = (generated_output + 1) / 2
    generated_output = generated_output * len(characters)
    generated_string = ''.join([characters[int(i)] for i in generated_output[0]])
    return generated_string

def generate_fake():
    human_generated_strings = df['merged_opcodes'].apply(lambda x: [ord(c) for c in x]).tolist()
    human_generated_strings = [torch.tensor(lst) for lst in human_generated_strings]
    human_generated_strings = pad_sequence(human_generated_strings, batch_first=True)
    generator = Generator(100, len(human_generated_strings[0]))
    generator.load_state_dict(torch.load('models/G_54c5741c-531c-11ee-b385-0aa07a8f18c1_0.32405543327331543.pkl'))
    fake_string = generate_fake_string(generator, 100, characters)
    print("Generated string:", fake_string[0:100])

#generate_fake()