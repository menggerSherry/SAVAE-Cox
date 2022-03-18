import torch.nn as nn
import torch
import torch.nn.functional as F


class CpyAndW(nn.Module):
    def __init__(self, num_heads, seq_length, sample_length):
        super(CpyAndW, self).__init__()
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.downsample = nn.Linear(seq_length,sample_length)
        self.ac = nn.LeakyReLU(inplace=True)
        for i in range(self.num_heads):
            setattr(self, 'weight_%d' % i, self.model(self.seq_length))

    def model(self, sample_length):
        model = nn.Sequential(
            nn.Linear(sample_length, sample_length),
            nn.LeakyReLU(inplace=True),
        )
        return model

    def forward(self, x):
        down_activate=self.ac(self.downsample(x))
        output = []
        for i in range(self.num_heads):
            output.append(getattr(self, 'weight_%d' % i)(down_activate).unsqueeze(1))
        # (batch,head,seq)
        output_value = torch.cat(output, dim=1)
        return output_value

class DownSample(nn.Module):
    def __init__(self,seq_length,channel):
        '''
        
        '''
        super(DownSample,self).__init__()
        self.channel=channel
        self.down_layer=nn.Conv1d(channel,channel,9,4,3)
        self.norm_layer=nn.BatchNorm1d(1)
        self.ac=nn.LeakyReLU(inplace=True)
    def forward(self,x):
        x=x.view(x.size(0),self.channel,x.size(1)).contiguous()
        downsample_out=self.ac(self.norm_layer(self.down_layer(x)))
        final_out=downsample_out.view(final_out.size(0),final_out.size(1)).contiguous()
        return final_out


class Attention(nn.Module):
    '''
    self attention
    '''
    def __init__(self, seq_length,dropout=True):
        '''
        sample_length
        '''
        super(Attention,self).__init__()
        self.seq_length=seq_length
        # self.sample_length=sample_length
        self.dropout=nn.Dropout(0.3)
        self.query=nn.Linear(seq_length,seq_length)
        self.key=nn.Linear(seq_length,seq_length)
        self.value=nn.Linear(seq_length,seq_length)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        # b,seq_length
        b,seq=x.size()
        q=self.query(x).view(b,-1,seq) # b,c,seq_length
        k=self.key(x).view(b,-1,seq)  # b,c,seq_length
        v=self.value(x).view(b,-1,seq) # b,c,seq_length
        
        attention=torch.bmm(q.permute(0,2,1).contiguous(),k) #b,seq,seq
        attention=self.softmax(attention)
        attention=self.dropout(attention)
        
        self_attention=torch.bmm(v,attention) #b,c,seq
        self_attention=self_attention.view(b,seq)

        out=self_attention + x
        return out


class MultiHeadAttention(nn.Module):
    '''multi-head attention'''

    def __init__(self, num_heads, seq_length, sample_length, dropout=True):
        super(MultiHeadAttention, self).__init__()
        self.seq_length = seq_length
        self.copy_and_weight = CpyAndW(num_heads, seq_length, sample_length)
        self.dropout = nn.Dropout(0.2)
        self.query = nn.Linear(sample_length, sample_length)
        self.key = nn.Linear(sample_length, sample_length)
        self.value = nn.Linear(sample_length, sample_length)
        self.final_layer = nn.Linear(num_heads, 1)

    def transpose_the_sequence(self, x):
        '''shape the sequence'''

        # (batch,head,seq) -> (batch,head,seq,1)
        new_x = x.unsqueeze(-1)
        return new_x

    def forward(self, x):
        input_x = self.copy_and_weight(x)

        q = self.query(input_x)
        k = self.key(input_x)
        v = self.value(input_x)

        q = self.transpose_the_sequence(q)
        k = self.transpose_the_sequence(k)
        v = self.transpose_the_sequence(v)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores /= self.seq_length ** 0.5
        attention_prob = F.softmax(attention_scores, dim=-1)
        # attention_prob = self.dropout(attention_prob)

        contex_layer = torch.matmul(attention_prob, v)
        contex_layer = self.dropout(contex_layer)
        # (batch,head,seq,1) -> (batch,seq,head)
        contex_layer = contex_layer.view(contex_layer.size(
            0), contex_layer.size(1), contex_layer.size(2)).contiguous()
        # out=self.final_layer(contex_layer)
        return F.leaky_relu(contex_layer)
