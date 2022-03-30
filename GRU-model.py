class myGRU(nn.Module):
  def __init__(self,input_size,hidden_size):
    super(myGRU,self).__init__()
    
    self.input_size = input_size
    self.output_size = output_size
    
    self.W_iz = nn.Linear(input_size,hidden_size)
    self.W_hz = nn.Linear(hidden_size,hidden_size)
    
    self.W_ir = nn.Linear(input_size,hidden_size)
    self.W_hr = nn.Linear(input_size,hidden_size)
    
    self.W_ih = nn.Linear(input_size,hidden_size)
    self.W_hh = nn.Linear(input_size,hidden_size)
    
  def forward(self,x,h_prev):
    z = torch.sigmoid(self.W_iz(x) + self.W_hz(h_prev))
    r = torch.sigmoid(self.W_ir(x) + self.W_hr(h_prev))
    g = torch.tanh(self.W_ih(x) + r*(self.W_hh(h_prev))
                   
    h_next = (1-z)*h_prev + z*g
                   
    return h_next
                
