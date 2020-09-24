import torch
import numpy as np
from grad_reverse_layer import grad_reverse


class SkipGramModel(torch.nn.Module):

  def __init__(self, vocab_size, aux_vocab_size, emb_dimension, aux_emb_dimension, aux_ratio, revgrad=False):
    super(SkipGramModel, self).__init__()
    self.vocab_size = vocab_size
    self.emb_dimension = emb_dimension
    self.aux_vocab_ize = aux_vocab_size
    self.aux_emb_dimension = aux_emb_dimension
    self.aux_ratio = aux_ratio
    
    self.u_embeddings = torch.nn.Embedding(vocab_size, emb_dimension)
    self.v_embeddings = torch.nn.Embedding(vocab_size, emb_dimension)
    self.aux_u_embeddings = torch.nn.Embedding(aux_vocab_size, emb_dimension)
    self.aux_v_embeddings = torch.nn.Embedding(aux_vocab_size, emb_dimension)

    initrange = 1.0 / self.emb_dimension
    torch.nn.init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
    torch.nn.init.constant_(self.v_embeddings.weight.data, 0)

    initrange = 1.0 / self.aux_emb_dimension
    torch.nn.init.uniform_(self.aux_u_embeddings.weight.data, -initrange, initrange)
    torch.nn.init.constant_(self.aux_v_embeddings.weight.data, 0)
    
    self.revgrad = revgrad
    
  def forward(self, pos_u, pos_v, neg_v, aux_pos_u, aux_pos_v, aux_neg_v):
    emb_u = self.u_embeddings(pos_u)
    emb_v = self.v_embeddings(pos_v)
    emb_neg_v = self.v_embeddings(neg_v)
    
    aux_emb_u = self.aux_u_embeddings(aux_pos_u)
    aux_emb_v = self.aux_v_embeddings(aux_pos_v)
    aux_emb_neg_v = self.aux_v_embeddings(aux_neg_v)

    score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
    score = torch.clamp(score, max=10, min=-10)
    score = -torch.nn.functional.logsigmoid(score)
    
    aux_score = torch.sum(torch.mul(aux_emb_u, aux_emb_v), dim=1)
    aux_score = torch.clamp(aux_score, max=10, min=-10)
    aux_score = -torch.nn.functional.logsigmoid(aux_score)

    neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
    neg_score = torch.clamp(neg_score, max=10, min=-10)
    neg_score = -torch.sum(torch.nn.functional.logsigmoid(-neg_score), dim=1)
    
    aux_neg_score = torch.bmm(aux_emb_neg_v, aux_emb_u.unsqueeze(2)).squeeze()
    aux_neg_score = torch.clamp(aux_neg_score, max=10, min=-10)
    aux_neg_score = -torch.sum(torch.nn.functional.logsigmoid(-aux_neg_score), dim=1)

    primary_x = torch.mean(score + neg_score)
    aux_x = torch.mean(aux_score + aux_neg_score)
    
    if self.revgrad:
        aux_x = grad_reverse(aux_x, self.aux_ratio)
    
    return primary_x, aux_x

  def save_embedding(self, id2word, file_name):
    embedding = self.u_embeddings.weight.cpu().data.numpy()
    with open(file_name, 'w') as f:
      f.write('%d %d\n' % (len(id2word), self.emb_dimension))
      for wid, w in id2word.items():
        e = ' '.join(map(lambda x: str(x), embedding[wid]))
        f.write('%s %s\n' % (w, e))
        
        
class LogitSGNSModel(torch.nn.Module):

  def __init__(self, vocab_size, aux_vocab_size, emb_dimension, aux_emb_dimension, epsilon, aux_ratio, revgrad=False):
    super(LogitSGNSModel, self).__init__()
    self.vocab_size = vocab_size
    self.emb_dimension = emb_dimension
    self.u_embeddings = torch.nn.Embedding(vocab_size, emb_dimension)
    self.v_embeddings = torch.nn.Embedding(vocab_size, emb_dimension)
    self.aux_ratio = aux_ratio
    
    self.aux_vocab_size = aux_vocab_size
    self.aux_emb_dimension = aux_emb_dimension
    self.aux_u_embeddings = torch.nn.Embedding(aux_vocab_size, aux_emb_dimension)
    self.aux_v_embeddings = torch.nn.Embedding(aux_vocab_size, aux_emb_dimension)
    
    self.eps = epsilon

    initrange = 1.0 / np.sqrt(self.emb_dimension)
    torch.nn.init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
    torch.nn.init.uniform_(self.v_embeddings.weight.data, -initrange, initrange)
    
    initrange = 1.0 / np.sqrt(self.aux_emb_dimension)
    torch.nn.init.uniform_(self.aux_u_embeddings.weight.data, -initrange, initrange)
    torch.nn.init.uniform_(self.aux_v_embeddings.weight.data, -initrange, initrange)
    
    self.revgrad = revgrad

  def forward(self, pos_u, pos_v, neg_v, aux_pos_u, aux_pos_v, aux_neg_v):
    emb_u = self.u_embeddings(pos_u)
    emb_v = self.v_embeddings(pos_v)
    emb_neg_v = self.v_embeddings(neg_v)
    
    aux_emb_u = self.aux_u_embeddings(aux_pos_u)
    aux_emb_v = self.aux_v_embeddings(aux_pos_v)
    aux_emb_neg_v = self.aux_v_embeddings(aux_neg_v)

    score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
    score = torch.clamp(score, min=self.eps, max=1-self.eps)
    score = -torch.log(score)
    
    aux_score = torch.sum(torch.mul(aux_emb_u, aux_emb_v), dim=1)
    aux_score = torch.clamp(aux_score, min=self.eps, max=1-self.eps)
    aux_score = -torch.log(aux_score)

    neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
    neg_score = torch.clamp(neg_score, min=self.eps, max=1-self.eps)
    neg_score = -torch.sum(torch.log(1-neg_score), dim=1)
    
    aux_neg_score = torch.bmm(aux_emb_neg_v, aux_emb_u.unsqueeze(2)).squeeze()
    aux_neg_score = torch.clamp(aux_neg_score, min=self.eps, max=1-self.eps)
    aux_neg_score = -torch.sum(torch.log(1-aux_neg_score), dim=1)
    
    primary_x = torch.mean(score + neg_score)
    aux_x = torch.mean(aux_score + aux_neg_score)
    
    if self.revgrad:
        aux_x = grad_reverse(aux_x, self.aux_ratio)

    return primary_x, aux_x

  def save_embedding(self, id2word, file_name):
    embedding = self.u_embeddings.weight.cpu().data.numpy()
    with open(file_name, 'w') as f:
      f.write('%d %d\n' % (len(id2word), self.emb_dimension))
      for wid, w in id2word.items():
        e = ' '.join(map(lambda x: str(x), embedding[wid]))
        f.write('%s %s\n' % (w, e))
