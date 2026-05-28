import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        # Backward compatibility, but we will use save_checkpoint directly
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

def save_checkpoint(model, optimizer, is_best=False, filename='checkpoint.pth'):
    """
    Saves the model and optimizer state. 
    Also saves a 'best_model.pth' if is_best is True.
    """
    folder = './models'
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, filename)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    # Graceful save to prevent corruption (write to temp first, then rename)
    temp_path = file_path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, file_path)
    
    if is_best:
        best_path = os.path.join(folder, 'best_model.pth')
        torch.save(checkpoint, best_path)

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    """
    Loads the model and optimizer state from checkpoint if it exists.
    Returns True if successful, False otherwise.
    """
    file_path = os.path.join('./models', filename)
    if os.path.exists(file_path):
        try:
            checkpoint = torch.load(file_path, weights_only=False) # allow dict
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False
            
    # Fallback to old basic model format for backward compatibility
    old_path = os.path.join('./models', 'model.pth')
    if os.path.exists(old_path):
        try:
            model.load_state_dict(torch.load(old_path, weights_only=True))
            print("Notice: Loaded legacy model.pth format. Optimizer state resets.")
            return True
        except Exception as e:
            print(f"Failed to load old model format: {e}")
            return False
            
    return False

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
