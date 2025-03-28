import time
import torch
# from embedding_offload.embedding import SparseEmbedding
from embedding_offload.embedding_adamw import SparseEmbedding
def test_simple_classification(vocab_size=10000, embed_dim=500, num_classes=5):
    """
    使用嵌入层构建简单文本分类模型，测试是否能达到相似的训练效果
    """
    print("\n===== 简单分类测试 =====")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建简单的分类模型
    class StandardEmbeddingClassifier(torch.nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
            self.fc = torch.nn.Linear(embed_dim, num_classes)
            
        def forward(self, x):
            # 假设x是[batch_size]的单词索引
            embedded = self.embedding(x)
            return self.fc(embedded)
    
    class OffloadedEmbeddingClassifier(torch.nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes, device):
            super().__init__()
            self.embedding = SparseEmbedding(vocab_size, embed_dim, optimizer_params = {
                "lr": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.0001,
                "eps": 1e-8,
            },
            optim_device='cpu')
            self.fc = torch.nn.Linear(embed_dim, num_classes).to(device)
            
        def forward(self, x):
            embedded = self.embedding(x)
            return self.fc(embedded)
    
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 创建一些合成数据
    num_samples = 1000
    # 为每个样本生成一个单词索引和一个类别
    indices = torch.randint(0, vocab_size, (num_samples,))
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # 训练模型
    batch_size = 64
    epochs = 200
    
    off_model = OffloadedEmbeddingClassifier(vocab_size, embed_dim, num_classes, device)
    # 输出每个参数的大小
    print("Offloaded Model Parameters:")
    for name, param in off_model.named_parameters():
        print(f"{name}: {param.size()}")
    off_optimizer = torch.optim.AdamW(off_model.parameters(), lr=0.001, weight_decay=0.0001)
    print("Training Offloaded Model...")
    torch.cuda.synchronize()
    time_start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        
        # 打乱数据
        perm = torch.randperm(num_samples)
        indices_shuffled = indices[perm]
        labels_shuffled = labels[perm]
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices_shuffled[i:i+batch_size].to(device)
            batch_labels = labels_shuffled[i:i+batch_size].to(device)
            
            # offload模型训练
            off_optimizer.zero_grad()
            outputs = off_model(batch_indices)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            off_optimizer.step()
            off_model.embedding.apply_gradients()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            epoch_loss += loss.item()
        
        accuracy = 100.0 * correct / num_samples
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    torch.cuda.synchronize()
    time_end = time.time()
    print(f"Training Time: {time_end - time_start:.2f}s")
    del off_model, off_optimizer
    

    std_model = StandardEmbeddingClassifier(vocab_size, embed_dim, num_classes).to(device)
    std_optimizer = torch.optim.AdamW(std_model.parameters(), lr=0.001, weight_decay=0.0001)
    print("Training Standard Model...")
    torch.cuda.synchronize()
    time_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        
        # 打乱数据
        perm = torch.randperm(num_samples)
        indices_shuffled = indices[perm]
        labels_shuffled = labels[perm]
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices_shuffled[i:i+batch_size].to(device)
            batch_labels = labels_shuffled[i:i+batch_size].to(device)
            
            # 标准模型训练
            std_optimizer.zero_grad()
            outputs = std_model(batch_indices)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            std_optimizer.step()

            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            epoch_loss += loss.item()
        
        accuracy = 100.0 * correct / num_samples
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    torch.cuda.synchronize()
    time_end = time.time()
    print(f"Training Time: {time_end - time_start:.2f}s")
    del std_model, std_optimizer
    
if __name__ == "__main__":
    test_simple_classification()