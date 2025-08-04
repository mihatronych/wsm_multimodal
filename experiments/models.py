from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn

class Conv1DReducer(BaseEstimator, TransformerMixin):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=3, out_features=64):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.out_features = out_features
        self.model = None

    def fit(self, X, y=None):
        input_len = X.shape[1]
        self.model = nn.Sequential(
            nn.Conv1d(1, self.out_channels, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.out_channels * input_len, self.out_features)
        )
        return self

    def transform(self, X):
        with torch.no_grad():
            x_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)
            return self.model(x_tensor).numpy()

from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
import copy

class Conv1DAttentionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=None, num_classes=2, lr=1e-3,
                 early_stopping_patience=10, early_stopping_delta=1e-4):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        
        self.model = None
        self.attn = None
        self.classifier = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None

        if input_size is not None:
            self._build_model(input_size)

    def _build_model(self, input_size):
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(32 * input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + 
            list(self.attn.parameters()) + 
            list(self.classifier.parameters()), lr=self.lr
        )

    def set_input_size(self, input_size):
        self._build_model(input_size)

    def fit(self, X, y):
        if self.model is None or self.input_size != X.shape[1]:
            self.set_input_size(X.shape[1])

        X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        best_loss = float('inf')
        best_state = None
        epochs_no_improve = 0

        for epoch in tqdm(range(100), desc="Training epochs"):
            self.model.train()
            x_conv = self.model(X_tensor)
            x_conv = x_conv.permute(0, 2, 1)

            attn_out, _ = self.attn(x_conv, x_conv, x_conv)
            flat = attn_out.reshape(attn_out.size(0), -1)
            logits = self.classifier(flat)

            loss = self.loss_fn(logits, y_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if loss.item() < best_loss - self.early_stopping_delta:
                best_loss = loss.item()
                best_state = {
                    'model': copy.deepcopy(self.model.state_dict()),
                    'attn': copy.deepcopy(self.attn.state_dict()),
                    'classifier': copy.deepcopy(self.classifier.state_dict())
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
                    break

        # Restore best state
        if best_state is not None:
            self.model.load_state_dict(best_state['model'])
            self.attn.load_state_dict(best_state['attn'])
            self.classifier.load_state_dict(best_state['classifier'])

        return self

    def predict(self, X):
        if self.model is None or self.input_size != X.shape[1]:
            raise RuntimeError("Model is not initialized with the correct input size.")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)
            x_conv = self.model(X_tensor)
            x_conv = x_conv.permute(0, 2, 1)
            attn_out, _ = self.attn(x_conv, x_conv, x_conv)
            flat = attn_out.reshape(attn_out.size(0), -1)
            logits = self.classifier(flat)
            return torch.argmax(logits, dim=1).numpy()

from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn

from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
import copy

class Conv1DClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=None, num_classes=2, lr=1e-3,
                 early_stopping_patience=10, early_stopping_delta=1e-4):
        self.input_size = input_size
        self.num_classes = num_classes
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

        self.model = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None

        if input_size is not None:
            self._build_model(input_size)

    def _build_model(self, input_size):
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * input_size, self.num_classes)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def set_input_size(self, input_size):
        """Call this to set or update the input size and rebuild the model."""
        self._build_model(input_size)

    def fit(self, X, y):
        if self.model is None or self.input_size != X.shape[1]:
            self.set_input_size(X.shape[1])

        X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in tqdm(range(100), desc="Training Conv1DClassifier"):
            self.model.train()
            logits = self.model(X_tensor)
            loss = self.loss_fn(logits, y_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if loss.item() < best_loss - self.early_stopping_delta:
                best_loss = loss.item()
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X):
        if self.model is None or self.input_size != X.shape[1]:
            raise RuntimeError("Model is not initialized with the correct input size.")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)
            logits = self.model(X_tensor)
            return torch.argmax(logits, dim=1).numpy()