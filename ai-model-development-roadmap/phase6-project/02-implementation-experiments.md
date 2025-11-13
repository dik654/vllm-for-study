# Week 15-16: Implementation & Experiments

## 🎯 목표

**설계를 실제로 구현하고 체계적인 실험 수행하기**

---

## 💻 Week 15: 모델 구현

### Day 1-3: 핵심 모델 구현

#### 1. 개발 순서

**Bottom-up Approach**:
```
1. Basic building blocks (layers, modules)
2. Combine into larger components
3. Assemble complete model
4. Test each step!
```

#### 2. 구현 체크리스트

**Phase 1: Forward Pass**
```python
# models/my_model.py
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Define all layers
        self.layer1 = ...
        self.layer2 = ...

    def forward(self, x):
        # 2. Define forward pass
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Test forward pass
model = MyModel(config)
x = torch.randn(2, 3, 224, 224)  # Dummy input
output = model(x)
print(f"Output shape: {output.shape}")  # Check!
```

**Phase 2: Backward Pass**
```python
# Test gradient flow
loss = output.mean()
loss.backward()

# Check gradients
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"WARNING: {name} has no gradient!")
    else:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
```

**Phase 3: Shape Validation**
```python
def test_shapes():
    """Test all intermediate shapes"""
    model = MyModel(config)
    x = torch.randn(batch_size, ...)

    # Hook to capture intermediate outputs
    activations = {}
    def hook(name):
        def fn(module, input, output):
            activations[name] = output.shape
        return fn

    # Register hooks
    for name, module in model.named_modules():
        module.register_forward_hook(hook(name))

    # Forward
    output = model(x)

    # Print all shapes
    for name, shape in activations.items():
        print(f"{name}: {shape}")

test_shapes()
```

#### 3. Unit Tests

```python
# tests/test_model.py
import pytest
import torch
from models.my_model import MyModel

def test_forward_shape():
    """Test output shape is correct"""
    model = MyModel(config)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    assert output.shape == (2, num_classes)

def test_backward():
    """Test backward pass works"""
    model = MyModel(config)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    loss = output.mean()
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None

def test_consistency():
    """Test model is deterministic"""
    model = MyModel(config)
    model.eval()

    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)

    assert torch.allclose(output1, output2)

# Run tests
pytest.main([__file__])
```

---

### Day 4-5: 훈련 파이프라인

#### 1. Data Loader

```python
# data/dataset.py
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None):
        self.data = load_data(data_path, split)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Apply transforms
        if self.transform:
            sample = self.transform(sample)

        return sample

# Create dataloaders
train_dataset = MyDataset('data/', split='train', transform=train_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

#### 2. Training Loop

```python
# training/trainer.py
import torch
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Logging
        wandb.init(project="my-project", config=config)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            inputs = batch['input'].to(self.config.device)
            targets = batch['target'].to(self.config.device)

            # Forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Log
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

            wandb.log({'train/loss': loss.item()})

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input'].to(self.config.device)
                targets = batch['target'].to(self.config.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                # Accuracy
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        wandb.log({
            'val/loss': avg_loss,
            'val/accuracy': accuracy
        })

        return avg_loss, accuracy

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            # Scheduler step
            self.scheduler.step()

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f'best_model.pt')

            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        wandb.finish()

    def save_checkpoint(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

---

### Day 6-7: Baseline 훈련

#### 1. 훈련 스크립트

```python
# scripts/train.py
import argparse
from models.baseline import BaselineModel
from training.trainer import Trainer
from data.dataset import create_dataloaders

def main(args):
    # Load data
    train_loader, val_loader = create_dataloaders(
        args.data_path,
        batch_size=args.batch_size
    )

    # Create model
    model = BaselineModel(args).to(args.device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train
    trainer = Trainer(model, train_loader, val_loader, args)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(args)
```

#### 2. 디버깅 체크리스트

**Loss가 감소하지 않으면**:
- [ ] Learning rate too high/low?
- [ ] Gradient flow 정상?
- [ ] Data loader 정상? (label 확인)
- [ ] Loss function 맞음?
- [ ] Overfitting 가능? (단일 배치로 테스트)

**Memory 부족**:
- [ ] Batch size 줄이기
- [ ] Gradient accumulation 사용
- [ ] Mixed precision (AMP)
- [ ] Model parallelism

---

## 🔬 Week 16: 실험 및 분석

### Day 1-3: Ablation Study

#### 실험 실행

```python
# Experiment configuration
experiments = {
    'baseline': {'comp_a': False, 'comp_b': False, 'comp_c': False},
    '+A': {'comp_a': True, 'comp_b': False, 'comp_c': False},
    '+B': {'comp_a': False, 'comp_b': True, 'comp_c': False},
    '+A+B': {'comp_a': True, 'comp_b': True, 'comp_c': False},
    'full': {'comp_a': True, 'comp_b': True, 'comp_c': True},
}

results = {}
for name, config in experiments.items():
    print(f"Running: {name}")

    # Train model with this configuration
    model = MyModel(config)
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

    # Evaluate
    metric = evaluate(model, test_loader)
    results[name] = metric

# Analyze
import pandas as pd
df = pd.DataFrame(results, index=['metric']).T
print(df)
df.to_csv('ablation_results.csv')
```

#### 결과 분석

```python
# Component contribution
contribution = {
    'A': results['+A'] - results['baseline'],
    'B': results['+B'] - results['baseline'],
    'C': results['full'] - results['+A+B'],
}

for comp, improvement in contribution.items():
    print(f"Component {comp}: +{improvement:.2f}%")
```

---

### Day 4-5: Hyperparameter Tuning

#### Grid Search

```python
from itertools import product

# Define search space
lr_values = [1e-5, 5e-5, 1e-4, 5e-4]
batch_sizes = [16, 32, 64]
weight_decays = [0.0, 0.01, 0.1]

best_metric = 0
best_config = None

for lr, bs, wd in product(lr_values, batch_sizes, weight_decays):
    config = {
        'lr': lr,
        'batch_size': bs,
        'weight_decay': wd
    }

    print(f"Testing: {config}")
    metric = train_and_evaluate(config)

    if metric > best_metric:
        best_metric = metric
        best_config = config

print(f"Best config: {best_config} with metric {best_metric}")
```

---

### Day 6-7: 최종 평가

#### 1. Test Set Evaluation

```python
# scripts/evaluate.py
def evaluate(model, test_loader, metrics):
    """Comprehensive evaluation"""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Compute metrics
    results = {}
    for metric_name, metric_fn in metrics.items():
        results[metric_name] = metric_fn(all_targets, all_preds)

    return results

# Run evaluation
metrics = {
    'accuracy': accuracy_score,
    'f1': lambda y, pred: f1_score(y, pred, average='weighted'),
    'precision': lambda y, pred: precision_score(y, pred, average='weighted'),
    'recall': lambda y, pred: recall_score(y, pred, average='weighted'),
}

test_results = evaluate(model, test_loader, metrics)
print("Test Results:")
for metric, value in test_results.items():
    print(f"  {metric}: {value:.4f}")
```

#### 2. Error Analysis

```python
def error_analysis(model, test_loader, num_samples=100):
    """Analyze failure cases"""
    model.eval()

    errors = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=-1)

            # Find errors
            incorrect = preds != targets
            if incorrect.any():
                indices = torch.where(incorrect)[0]
                for idx in indices:
                    errors.append({
                        'input': inputs[idx],
                        'target': targets[idx].item(),
                        'pred': preds[idx].item(),
                        'confidence': outputs[idx].max().item()
                    })

                if len(errors) >= num_samples:
                    break

    # Analyze patterns
    print(f"Total errors: {len(errors)}")

    # Confusion patterns
    from collections import Counter
    confusion = Counter([(e['target'], e['pred']) for e in errors])
    print("Most common confusions:")
    for (true, pred), count in confusion.most_common(10):
        print(f"  True: {true} → Pred: {pred} ({count} times)")

    return errors
```

#### 3. Visualization

```python
def visualize_results(model, test_data, num_samples=10):
    """Visualize predictions"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, sample in enumerate(test_data[:num_samples]):
        # Predict
        with torch.no_grad():
            pred = model(sample['input'].unsqueeze(0)).argmax().item()

        # Plot
        axes[i].imshow(sample['input'].permute(1, 2, 0))
        axes[i].set_title(f"True: {sample['target']}\nPred: {pred}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
```

---

## 📊 Final Report

### Report Structure

```markdown
# [Your Project Title]

## Abstract
- Problem
- Method
- Results (1-2 sentences)

## 1. Introduction
- Motivation
- Problem statement
- Contributions

## 2. Related Work
- Existing methods
- Their limitations
- How ours is different

## 3. Method
- Architecture overview (diagram!)
- Key components explained
- Novel contributions highlighted

## 4. Experiments

### 4.1 Experimental Setup
- Dataset
- Baselines
- Metrics
- Implementation details

### 4.2 Results
- Main results (table)
- Comparison with baselines
- Statistical significance

### 4.3 Ablation Study
- Component contributions
- Analysis

### 4.4 Analysis
- Error analysis
- Visualizations
- Computational cost

## 5. Conclusion
- Summary
- Limitations
- Future work

## References
```

---

## 🎓 Deliverables (Week 15-16)

- [ ] Complete implementation (tested, documented)
- [ ] Trained models (baseline + your model)
- [ ] Experimental results (tables, figures)
- [ ] Ablation study results
- [ ] Test set evaluation
- [ ] Error analysis
- [ ] Final report (markdown or LaTeX)
- [ ] Code repository (clean, README)
- [ ] Model checkpoints (HuggingFace Hub)

---

## 🎉 Congratulations!

**당신은 완전한 AI 연구 프로젝트를 완료했습니다!**

이제 다음을 할 수 있습니다:
- ✅ 논문 작성 및 제출
- ✅ 오픈소스 공유
- ✅ 포트폴리오 추가
- ✅ 취업/진학 지원

**The journey continues!** 🚀
