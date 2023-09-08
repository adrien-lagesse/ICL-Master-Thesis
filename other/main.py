import datetime
import torch
from models import GCN
from torch.utils.tensorboard import SummaryWriter

# Downloading the ESOL dataset from the MoleculeNet paper
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
data = MoleculeNet(root = "esol_dataset/", name ="ESOL")

# Parameters
IN_FEATURES = data.num_features
OUT_FEATURES = 1
EMBEDDING_SIZE = 256

LEARNING_RATE = 2e-3
EPOCHS = 5000
BATCH_SIZE = 256
SHUFFLE = True

EXPERIMENT_NAME = f"GCN_ESOL/{datetime.datetime.ctime(datetime.datetime.now())}"

# Setting up Tensorboard logs
writer = SummaryWriter(log_dir=f"logs/{EXPERIMENT_NAME}", comment='GCN_ESOL')

# GCN model for Graph Regression
model = GCN(num_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            embedding_size=EMBEDDING_SIZE)

# Putting everything on the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data.data.to(device)


# Training initialization
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loader = DataLoader(data, BATCH_SIZE, shuffle=SHUFFLE)

def train():
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        y_pred = model(batch.x.float(), batch.edge_index, batch.batch) 
        loss = torch.sqrt(loss_fn(y_pred, batch.y))       
        loss.backward()
        optimizer.step()

    return loss.data.detach()

# Training
for epoch in range(EPOCHS):
    loss = train()
    writer.add_scalar('Loss/train',loss,epoch)
    if epoch % 50 == 0:
      print(f"Epoch {epoch} ---- Train Loss: {loss}")