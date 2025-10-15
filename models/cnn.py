import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- CNN for MNIST ---------------- #

class CNN_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)      # 28x28 -> 24x24
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)     # 12x12 pooled -> 8x8
        self.pool = nn.MaxPool2d(2, 2)                    # pooling reduces size
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Keys for federated fine-tuning
        self.base_weight_keys = [
            'conv1.weight', 'conv1.bias',
            'conv2.weight', 'conv2.bias',
            'fc1.weight', 'fc1.bias',
        ]
        self.classifier_weight_keys = [
            'fc2.weight', 'fc2.bias',
        ]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        features = torch.flatten(x, 1)       # flatten for FC
        features_fc = F.leaky_relu(self.fc1(features))
        logits = self.fc2(features_fc)
        return features_fc, logits

    def feature2logit(self, features):
        return self.fc2(features)


# ---------------- Hypernetwork to generate CNN weights ---------------- #

class Hypernetwork_CNN_MNIST(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Hypernetwork_CNN_MNIST, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Generate CNN feature extractor weights
        self.c1_weights = nn.Linear(hidden_dim, 16 * 1 * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, 16)
        self.c2_weights = nn.Linear(hidden_dim, 32 * 16 * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 32)
        self.l1_weights = nn.Linear(hidden_dim, 128 * 32 * 4 * 4)
        self.l1_bias = nn.Linear(hidden_dim, 128)

    def forward(self, client_embedding):
        features = self.mlp(client_embedding)
        weights = {
            "conv1.weight": self.c1_weights(features).view(16, 1, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(32, 16, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(128, 32 * 4 * 4),
            "fc1.bias": self.l1_bias(features).view(-1),
        }
        return weights


# ---------------- CNN with Hypernetwork ---------------- #

class CNN_MNIST_Hyper(nn.Module):
    def __init__(self, args):
        super(CNN_MNIST_Hyper, self).__init__()
        self.target_model = CNN_MNIST(num_classes=args.num_classes)

        # Use classifier keys from target_model for fine-tuning
        self.classifier_weight_keys = self.target_model.classifier_weight_keys

        self.client_embedding = nn.Embedding(num_embeddings=1, embedding_dim=args.embed_dim)
        self.hypernetwork = Hypernetwork_CNN_MNIST(
            embedding_dim=args.embed_dim, hidden_dim=args.hidden_dim
        )

    def forward(self, x):
        # Generate client-specific weights
        client_embedding = self.client_embedding(
            torch.tensor([0], dtype=torch.long, device=x.device)
        )
        weights = self.hypernetwork(client_embedding)
        self.target_model.load_state_dict(weights, strict=False)

        features, logits = self.target_model(x)
        return features, logits

    def predict(self, x):
        _, logits = self.target_model(x)
        return logits

    def generate_weight(self):
        device = next(self.parameters()).device
        client_embedding = self.client_embedding(
            torch.tensor([0], dtype=torch.long, device=device)
        )
        weights = self.hypernetwork(client_embedding)
        return weights


