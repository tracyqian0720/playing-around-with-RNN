class myLSTM(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(myLSTM, self).__init__()
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    
    self.W_if = nn.Linear(input_size,hidden_size)
    self.W_hf = nn.Linear(hidden_size,hidden_size)
    
    self.W_ii = nn.Linear(input_size,hidden_size)
    self.W_hi = nn.Linear(hidden_size,hidden_size)
    
    self.W_ic = nn.Linear(input_size,hidden_size)
    self.W_hc = nn.Linear(hidden_size,hidden_size)
    
    self.W_io = nn.Linear(input_size,hidden_size)
    self.W_ho = nn.Linear(hidden_size,hidden_size)
   
  def forward(self, x, h_prev, c_prev):
    
    f = torch.sigmoid(self.W_if(x) + self.W_hf(h_prev))
    i = torch.sigmoid(self.W_ii(x) + self.W_hi(h_prev))
    c = torch.tanh(self.W_ic(x) + self.W_hc(h_prev))
    o = torch.tanh(self.W_io(x) + self.W_ho(h_prev))
    
    c_next = f*c_prev + i*c
    h_next = o*torch.tanh(c_new)
    
    return c_next,h_next
