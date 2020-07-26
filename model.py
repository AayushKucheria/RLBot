import torch
import torch.nn as nn  # Provides us with many classes and parameters to implement a neural network
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# Contains implementations of Actor-Critic (AC) Algorithms. The Critic estimates the value function and the Actor
# follows the critic's suggestion.
# They canlearn directly from raw exp without an env model - these methods are called Temporal Difference Methods.
# AC Methods are different from TD as they've a separate memory structure to represent policy (Actor) - used to select actions.
# In a way, the critic acts like an error estimator. O/p = Scalar (+ve if good -ve if bad)
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# RecurrentACModel: Actor-Critic Method implementing Recurrent Neural Networks.
# RNN: Normal NN learn during training, and RNN additionally takes prior o/p as i/p (context) as a hidden vector.
#   Normal NN contain fixed no. of i/ps and o/ps. RNN contains a independent series and they're linked by the "context".
#   The context is updated based on i/p every iteration.
class ACModel(nn.Module, torch_ac.RecurrentACModel):  # nn.Module: Base class for Neural Networks,
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(  # A sequential container, like a list.
            nn.Conv2d(3, 16, (2, 2)),  # TODO: 2D convolution over an i/p signal composed of several i/p planes
            nn.ReLU(),  # TODO: Rectified Linear Unit Function
            nn.MaxPool2d((2, 2)),  # TODO: 2D max pooling over an i/p signal composed of several i/p planes
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]  # TODO: What does obs space contain exactly?
        m = obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64  # Using Floor Division

        # Define memory
        if self.use_memory:
            # Params: (input size, hidden size)
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)  # Apply a LSTM cell to an input sequence

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)  # Store word embedding's and retrieve them using indices TODO: What are embeddings?
            self.text_embedding_size = 128

            # Parameters -> (no. of expected features in i/p x, no. of expected features in hidden state h, batch_first)
            # Batch_first: If true, then i/p and o/p tensors are provided as (batch, seq, feature)
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)  # Apply a multi-layer gated recurrent unit (GRU) RNN to an i/p sequence

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),  # Apply a linear transformation: y = xA^t + b. (size of i/p sample, size of o/p sample, bias)
            nn.Tanh(),  # Activation function tanh(x)
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    # I/Ps N observations and M memories. Returns distribution, values, and memories.
    def forward(self, obs, memory):  # TODO: What's happening here?
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
