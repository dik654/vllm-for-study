# Day 7: Training Pipeline

## 🎯 목표

**Production-ready 훈련 파이프라인 구축하기**

```python
# 이 장에서 배울 것:
- Trainer API vs Native PyTorch
- 분산 훈련 (DeepSpeed, FSDP)
- 메모리 최적화 (Gradient Checkpointing, Flash Attention)
- 로깅 및 모니터링 (W&B, TensorBoard)
```

---

## 🏗️ Part 1: Trainer API

### 1.1 Basic Training

**HuggingFace Trainer는 모든 것을 간단하게!**

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# 1. Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 2. Load dataset
dataset = load_dataset('glue', 'sst2')

# 3. Preprocess
def preprocess(examples):
    return tokenizer(
        examples['sentence'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

tokenized_dataset = dataset.map(preprocess, batched=True)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir='./results',

    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,

    # Evaluation
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',

    # Logging
    logging_dir='./logs',
    logging_steps=100,

    # Hardware
    fp16=True,  # Mixed precision
    dataloader_num_workers=4
)

# 5. Metric
from datasets import load_metric

def compute_metrics(eval_pred):
    metric = load_metric('accuracy')
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    compute_metrics=compute_metrics
)

# 7. Train!
trainer.train()

# 8. Evaluate
results = trainer.evaluate()
print(results)

# 9. Save
trainer.save_model('./final_model')
```

---

### 1.2 Custom Training Loop

**더 세밀한 제어가 필요하면 Native PyTorch**

```python
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# Model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = model.cuda()

# DataLoader
train_loader = DataLoader(
    tokenized_dataset['train'],
    batch_size=16,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    tokenized_dataset['validation'],
    batch_size=32,
    num_workers=4
)

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01
)

# Scheduler
num_training_steps = len(train_loader) * 3  # 3 epochs
num_warmup_steps = num_training_steps // 10

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Training loop
for epoch in range(3):
    # Train
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        # Move to device
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['label'].cuda()

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Evaluate
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['label'].cuda()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            eval_loss += outputs.loss.item()

            # Accuracy
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_eval_loss = eval_loss / len(val_loader)
    accuracy = correct / total

    print(f'Epoch {epoch+1}:')
    print(f'  Train Loss: {avg_train_loss:.4f}')
    print(f'  Eval Loss: {avg_eval_loss:.4f}')
    print(f'  Accuracy: {accuracy:.4f}')
```

---

## 🚀 Part 2: 분산 훈련

### 2.1 DataParallel (DP)

**단순하지만 비효율적**

```python
import torch.nn as nn

# Wrap model
model = nn.DataParallel(model)

# Train normally
# → 자동으로 여러 GPU 사용
```

**문제**:
- GPU 0에 병목 (gradient 집계)
- 메모리 비효율적

---

### 2.2 DistributedDataParallel (DDP)

**효율적인 multi-GPU 훈련**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp(rank, world_size):
    """Initialize DDP"""
    dist.init_process_group(
        backend='nccl',  # NVIDIA GPUs
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    """Training function for each process"""
    # Setup
    setup_ddp(rank, world_size)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Sampler (각 GPU가 다른 데이터)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=4
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop
    for epoch in range(3):
        sampler.set_epoch(epoch)  # Important!

        for batch in train_loader:
            # Move to device
            input_ids = batch['input_ids'].to(rank)
            labels = batch['label'].to(rank)

            # Forward
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Cleanup
    cleanup_ddp()

# Launch with torchrun
# torchrun --nproc_per_node=4 train.py
```

**HuggingFace Trainer로 간단히**:

```python
training_args = TrainingArguments(
    # ...
    # DDP 자동 활성화!
)

# Launch:
# python -m torch.distributed.launch --nproc_per_node=4 train.py
```

---

### 2.3 DeepSpeed

**Microsoft의 최적화 라이브러리**

**ZeRO (Zero Redundancy Optimizer)**:
- Stage 1: Optimizer state sharding
- Stage 2: + Gradient sharding
- Stage 3: + Parameter sharding

```python
# deepspeed_config.json
{
    "train_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,  # ZeRO Stage 2
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

**Trainer 통합**:

```python
training_args = TrainingArguments(
    output_dir='./results',
    deepspeed='./deepspeed_config.json',
    # ...
)

# Launch:
# deepspeed train.py
```

---

### 2.4 FSDP (Fully Sharded Data Parallel)

**PyTorch 네이티브 (2.0+)**

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

training_args = TrainingArguments(
    output_dir='./results',
    fsdp='full_shard auto_wrap',  # FSDP 활성화
    fsdp_config={
        'min_num_params': 1e6,  # Shard 최소 크기
        'transformer_layer_cls_to_wrap': 'BertLayer'  # Wrap할 layer
    },
    # ...
)
```

---

## 💾 Part 3: 메모리 최적화

### 3.1 Mixed Precision (FP16/BF16)

**메모리 절반, 속도 2배!**

```python
# FP16 (older GPUs)
training_args = TrainingArguments(
    fp16=True,
    # ...
)

# BF16 (A100, H100)
training_args = TrainingArguments(
    bf16=True,
    # ...
)

# Manual (PyTorch AMP)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()

    # Forward with autocast
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss

    # Backward with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

### 3.2 Gradient Accumulation

**큰 effective batch size with 작은 메모리**

```python
# Effective batch size = per_device_batch_size × gradient_accumulation_steps × num_gpus

training_args = TrainingArguments(
    per_device_train_batch_size=4,  # Fits in memory
    gradient_accumulation_steps=8,  # Effective batch = 32
    # ...
)

# Manual
accumulation_steps = 8
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps  # Scale loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### 3.3 Gradient Checkpointing

**메모리 절약 (시간 약간 증가)**

```python
# HuggingFace
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    gradient_checkpointing=True,
    # ...
)

# 원리: Forward 중 일부 activation 버림 → Backward 시 재계산
# 메모리: 50% 절약
# 시간: 20% 증가
```

---

### 3.4 Flash Attention

**Attention 메모리/속도 최적화**

```python
# Install
# pip install flash-attn

# Enable
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'gpt2',
    attn_implementation='flash_attention_2',  # Flash Attention 2
    torch_dtype=torch.float16
)

# Benefits:
# - 2-4x faster
# - 10x less memory
# - Longer sequences (10k+ tokens)
```

---

## 📊 Part 4: 로깅 및 모니터링

### 4.1 Weights & Biases (W&B)

```python
# Install
# pip install wandb

import wandb

# Initialize
wandb.init(
    project='my-project',
    name='bert-finetuning',
    config={
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 3
    }
)

# HuggingFace Trainer 통합
training_args = TrainingArguments(
    report_to='wandb',  # 자동 로깅!
    # ...
)

# Manual logging
for epoch in range(3):
    for step, batch in enumerate(train_loader):
        # ... training ...

        wandb.log({
            'train/loss': loss.item(),
            'train/learning_rate': scheduler.get_last_lr()[0],
            'train/epoch': epoch
        })

# Log artifacts
wandb.log_artifact(model_path, type='model')

# Finish
wandb.finish()
```

---

### 4.2 TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

# Initialize
writer = SummaryWriter(log_dir='./logs')

# HuggingFace Trainer
training_args = TrainingArguments(
    logging_dir='./logs',
    report_to='tensorboard',
    # ...
)

# Manual logging
for epoch in range(3):
    for step, batch in enumerate(train_loader):
        # ... training ...

        writer.add_scalar('Loss/train', loss.item(), step)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], step)

# Log histograms
for name, param in model.named_parameters():
    writer.add_histogram(name, param, epoch)

# Close
writer.close()

# View:
# tensorboard --logdir=./logs
```

---

### 4.3 Custom Callbacks

```python
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    """Custom training callback"""

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} starting...")

    def on_step_end(self, args, state, control, **kwargs):
        # Check for NaN
        if state.log_history and 'loss' in state.log_history[-1]:
            loss = state.log_history[-1]['loss']
            if np.isnan(loss):
                print("WARNING: NaN loss detected!")
                control.should_training_stop = True

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"Evaluation metrics: {metrics}")

        # Early stopping (custom)
        if metrics.get('eval_accuracy', 0) > 0.95:
            print("Accuracy threshold reached! Stopping.")
            control.should_training_stop = True

# Use callback
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[CustomCallback()]
)
```

---

## 🔧 Part 5: 실전 팁

### 5.1 Resume Training

```python
# Trainer auto-resume
training_args = TrainingArguments(
    resume_from_checkpoint='./results/checkpoint-1000',
    # ...
)

# Manual checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

### 5.2 Learning Rate Finder

```python
from torch_lr_finder import LRFinder

# Model & Optimizer
model = ...
optimizer = AdamW(model.parameters(), lr=1e-7)

# LR Finder
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)

# Plot
lr_finder.plot()  # Shows loss vs LR
lr_finder.reset()  # Reset model & optimizer

# Suggested LR
suggested_lr = lr_finder.suggest_lr()
print(f"Suggested LR: {suggested_lr}")
```

---

### 5.3 Complete Training Template

```python
# train.py
import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset
import wandb

def main():
    # Config
    MODEL_NAME = 'bert-base-uncased'
    OUTPUT_DIR = './results'
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 3

    # W&B
    wandb.init(project='bert-classification')

    # Load
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset('glue', 'sst2')

    # Preprocess
    def preprocess(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length')

    tokenized = dataset.map(preprocess, batched=True)

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,

        # Evaluation
        evaluation_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',

        # Logging
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=100,
        report_to='wandb',

        # Optimization
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_num_workers=4,

        # DDP
        ddp_find_unused_parameters=False
    )

    # Metrics
    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)

        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted')
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate(tokenized['test'])
    print(f"Test results: {results}")

    # Save
    trainer.save_model(f'{OUTPUT_DIR}/final_model')

    # Upload to Hub (optional)
    # trainer.push_to_hub("my-model")

    wandb.finish()

if __name__ == '__main__':
    main()

# Launch:
# Single GPU: python train.py
# Multi GPU: torchrun --nproc_per_node=4 train.py
# DeepSpeed: deepspeed train.py --deepspeed ds_config.json
```

---

## 🎓 학습 목표

- [ ] Trainer API와 Native PyTorch 차이 이해
- [ ] DDP, DeepSpeed, FSDP 설정
- [ ] Mixed precision training 적용
- [ ] Gradient checkpointing 이해
- [ ] W&B/TensorBoard로 모니터링
- [ ] 완전한 훈련 파이프라인 구축

---

## 📊 성능 비교

| Method | Speed | Memory | Scalability |
|--------|-------|--------|-------------|
| Single GPU | 1x | 1x | ❌ |
| DataParallel | 1.5x | 1x | ⚠️ GPU 0 병목 |
| DDP | 3.8x | 1x | ✅ Linear |
| DeepSpeed ZeRO-2 | 3.5x | 0.5x | ✅ Linear |
| DeepSpeed ZeRO-3 | 3.2x | 0.3x | ✅ Linear |
| FSDP | 3.7x | 0.4x | ✅ Linear |

---

## 🎉 축하합니다!

**Phase 0 완료! HuggingFace 생태계를 완전히 마스터했습니다!**

이제 다음을 할 수 있습니다:
- ✅ Production-ready 훈련 파이프라인 구축
- ✅ 분산 훈련으로 대규모 모델 훈련
- ✅ 메모리 최적화로 GPU 효율 극대화
- ✅ 체계적인 실험 관리

---

## ⏭️ 다음 단계

👉 [Phase 1: Transformer 구현](../phase1-transformer/)

**이제 Transformer를 밑바닥부터 구현하며 내부 작동 원리를 완전히 이해합니다!** 🚀
