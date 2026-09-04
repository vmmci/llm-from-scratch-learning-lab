# 09 — Saving & Loading Models

## Overview

Training a neural network can take a significant amount of time and compute.

Once a model has learned useful parameters, we need a way to preserve them so
that we can:

- reuse the model later
- resume training
- run inference without retraining
- create checkpoints during long training runs
- move trained weights between environments

The core workflow is:

```text
Train model
    ↓
Save parameters
    ↓
Close / restart environment
    ↓
Recreate model architecture
    ↓
Load parameters
    ↓
Continue inference or training
```

---

# Why Saving Matters

A trained model is valuable because of the parameter values it has learned.

If training is interrupted and the parameters were never saved, the learned
state can be lost.

Saving therefore separates:

```text
Model architecture
        +
Learned parameters
```

from the temporary Python session in which training occurred.

---

# Saving a PyTorch Model

The workshop uses PyTorch's `state_dict()`.

```python
torch.save(
    model.state_dict(),
    "/mnt/DON-v1.pth"
)
```

The important object being saved is:

```python
model.state_dict()
```

This contains the model's learned parameter tensors.

Conceptually:

```text
Model
  ↓
state_dict()
  ↓
weights + biases
  ↓
.pth file
```

---

# What `state_dict()` Represents

A neural-network architecture defines the structure of the model.

For example:

```text
Linear layer
ReLU
Linear layer
```

The `state_dict` stores the learned state associated with the trainable parts of
that architecture.

This means the saved file is not the same thing as the Python class definition.

A useful distinction is:

```text
Architecture
    ↓
Python model class

Learned state
    ↓
state_dict
```

Both are needed when loading the model using this workflow.

---

# Loading a Saved Model

The workshop recreates the model first:

```python
new_model = Mod5MLP()
```

and then loads the saved parameters:

```python
new_model.load_state_dict(
    torch.load("/mnt/DON-v1.pth")
)
```

The sequence is:

```text
Create architecture
       ↓
Load saved state_dict
       ↓
Parameters restored
       ↓
Model ready to use
```

---

# Architecture Must Match

The newly created model must be compatible with the saved parameter tensors.

Conceptually:

```text
Saved weights
      ↓
expected tensor shapes

New architecture
      ↓
must expose matching parameter shapes
```

If the architecture changes, the saved state may no longer load directly.

This is why model structure and checkpoint version need to be tracked together.

---

# Verifying the Loaded Model

After loading, the workshop verifies that the new model still produces the
expected behavior.

Conceptually:

```text
Original trained model
       ↓
save
       ↓
new Python model
       ↓
load parameters
       ↓
same learned behavior
```

This validation step is important because successfully reading a file is not by
itself proof that the correct model state was restored.

---

# Saving Locations

The workshop demonstrates multiple storage locations.

## Local Environment

A model can be saved directly to the local filesystem used by the runtime.

Example:

```python
torch.save(
    model.state_dict(),
    "/mnt/DON-v1.pth"
)
```

---

## Google Drive / Persistent Storage

In notebook environments such as Colab, the runtime itself can be temporary.

Persistent storage allows checkpoints to survive when the runtime is restarted.

The workshop therefore also demonstrates saving model files to Google Drive.

Conceptually:

```text
Temporary Colab runtime
       ↓
training
       ↓
checkpoint
       ↓
Google Drive
       ↓
persistent storage
```

---

# Loading from External Storage

The workshop also demonstrates restoring model weights from external storage or
a URL.

The workflow is still conceptually the same:

```text
Locate checkpoint
      ↓
Load checkpoint bytes
      ↓
Restore state_dict
      ↓
Use model
```

The physical storage location changes, but the model-loading logic remains
similar.

---

# Model File Formats

The slides introduce several model-storage and deployment formats.

They serve different purposes.

---

## 1 — PTH

`.pth` is the format used throughout this workshop for PyTorch checkpoints.

The workshop uses it because it integrates directly with:

```python
torch.save(...)
torch.load(...)
```

and PyTorch `state_dict` objects.

The GPT-style architecture decision for this workshop is:

```text
Use .pth files to save and load model weights.
```

---

# Evolution of Model File Formats

The slides present a progression of model formats used in different parts of the
ML ecosystem.

```text
PTH
 ↓
SafeTensors
 ↓
ONNX
 ↓
GGUF
 ↓
CoreML
```

These formats are not simply newer versions of the same thing.

They address different storage, interoperability, inference, and deployment
requirements.

---

## PTH

Workshop role:

```text
PyTorch model storage
```

Advantages in this project:

- simple PyTorch integration
- convenient for checkpoints
- works directly with `state_dict`

The workshop uses `.pth` as its main checkpoint format.

---

## SafeTensors

The slides describe SafeTensors as a format designed for safer tensor storage.

It avoids arbitrary code execution behavior associated with pickle-based model
serialization.

It is especially common in modern model-sharing ecosystems.

---

## ONNX

ONNX focuses on interoperability.

The general workflow is:

```text
Model framework
      ↓
Export
      ↓
ONNX
      ↓
Compatible inference/runtime systems
```

Its purpose is to help models move between frameworks and inference
environments.

---

## GGUF

GGUF is presented as an inference-oriented format.

It is commonly associated with optimized local inference and quantized model
weights.

The focus is different from a training checkpoint.

---

## CoreML

CoreML targets deployment in the Apple ecosystem.

The slides present it as a mobile/device-oriented deployment format.

Again, its purpose is different from the `.pth` checkpoint used during model
development.

---

# Training Checkpoints

Saving a model only once at the end of training is risky for long training
runs.

Instead, training can periodically create **checkpoints**.

Conceptually:

```text
Training
  ↓
Step 1,000
  ↓
Checkpoint

Training continues
  ↓
Step 5,000
  ↓
Checkpoint

Training continues
  ↓
Step 10,000
  ↓
Checkpoint
```

If training stops unexpectedly, the latest checkpoint can be loaded and
training can continue from a later point rather than restarting completely.

---

# Workshop Checkpoint Strategy

The slides specify the GPT-2-style workshop decision:

```text
Checkpoint:
latest .pth every 100 pretraining steps
```

The exact number is a workshop architecture choice.

The important principle is:

> Long-running training should save state periodically.

---

# Versioned Checkpoints

Instead of repeatedly overwriting one file, checkpoint names can encode training
progress.

For example:

```text
workshop-v1-pretraining.pth
workshop-v1-step100.pth
workshop-v1-step200.pth
...
```

This allows different stages of training to be preserved.

The slide shows a checkpoints directory containing multiple model states of
different sizes and training stages.

---

# Workshop Folder Structure

The architecture decision shown in the slides stores model files in a workshop
folder.

Conceptually:

```text
/workshop/
    checkpoints/
        ...
    model files
```

The folder can live in persistent storage such as Google Drive when running in
Colab.

This creates separation between:

```text
code
data
checkpoints
model artifacts
```

---

# Saving vs Loading

These two operations perform opposite transformations.

## Saving

```text
Model in memory
      ↓
state_dict
      ↓
Serialized file
```

## Loading

```text
Serialized file
      ↓
state_dict
      ↓
Model in memory
```

Together:

```text
Train
 ↓
Save
 ↓
Stop session
 ↓
Load
 ↓
Continue
```

---

# Saving for Inference

A saved model can be loaded later without repeating training.

```text
Saved checkpoint
      ↓
Load model
      ↓
Input
      ↓
Inference
      ↓
Prediction
```

This is essential when a trained model is used as an application rather than as
a training experiment.

---

# Saving for Continued Training

Checkpoints can also support resumed training.

Conceptually:

```text
Training state
      ↓
checkpoint
      ↓
restart environment
      ↓
restore model
      ↓
continue training
```

In the basic workshop implementation, the focus is primarily on saving and
loading the model weights.

Later or larger training systems may preserve additional training state as
needed.

---

# Connection to Unit 08

Unit 08 established:

```text
Random parameters
      ↓
Training
      ↓
Backpropagation
      ↓
Optimizer
      ↓
Learned parameters
```

Unit 09 adds persistence:

```text
Learned parameters
      ↓
Save
      ↓
File
      ↓
Load
      ↓
Learned parameters restored
```

Without saving, all of the work performed by the optimizer exists only inside
the current runtime.

---

# Connection to Large Language Models

The same principle applies to LLM training.

At large scale:

```text
Pretraining
    ↓
millions / billions of parameter updates
    ↓
checkpoint
    ↓
persistent storage
```

A checkpoint may later be used for:

- continued pretraining
- evaluation
- inference
- fine-tuning
- experimentation
- deployment conversion

The fundamental idea remains the same as the small PyTorch example.

---

# Checkpoints as Model History

A sequence of checkpoints can also be viewed as snapshots of model development.

```text
Initial model
    ↓
Checkpoint A
    ↓
Checkpoint B
    ↓
Checkpoint C
    ↓
Final model
```

This allows training progress to be inspected or recovered at multiple stages.

---

# Architecture Decisions in This Workshop

The slides make several specific GPT-2-style decisions for the project.

```text
Format:
.pth

Storage:
Google Drive / workshop folder

Model saving:
state_dict checkpoints

Checkpoint frequency:
latest model approximately every 100 pretraining steps
```

These decisions are practical choices for the workshop rather than universal
rules for every ML system.

---

# Key Takeaways

1. Training results must be saved if they are expected to survive beyond the
   current runtime.

2. PyTorch's `state_dict()` contains the learned model state used in this
   workshop.

3. `torch.save()` serializes that state to a file.

4. `torch.load()` reads the saved state.

5. `load_state_dict()` restores the parameters into a compatible model
   architecture.

6. The model architecture must be recreated before loading a `state_dict`.

7. Loading should be verified by checking that the restored model behaves as
   expected.

8. Persistent storage is especially important in temporary notebook
   environments such as Colab.

9. `.pth` is the checkpoint format selected for this workshop.

10. SafeTensors, ONNX, GGUF, and CoreML serve different model-storage or
    deployment purposes.

11. Long-running training should create periodic checkpoints.

12. Checkpoints make interrupted training recoverable.

13. Multiple checkpoint files can preserve different stages of model training.

14. Saving and loading separate learned model state from the lifetime of a
    Python process.

15. The same checkpoint principle used for a small neural network scales to
    large language-model training.

---

# Final Mental Model

Before saving:

```text
Architecture
    +
Learned parameters
    ↓
Model exists in memory
```

Saving converts the learned state into a persistent artifact:

```text
Model
  ↓
state_dict
  ↓
.pth checkpoint
```

Loading reverses the process:

```text
Model architecture
        +
Saved state_dict
        ↓
Restored trained model
```

The complete lifecycle is therefore:

\[
\boxed{
\text{Build}
\rightarrow
\text{Train}
\rightarrow
\text{Save}
\rightarrow
\text{Load}
\rightarrow
\text{Reuse}
}
\]

Saving and loading turn model training from a temporary runtime activity into a
persistent and reusable artifact.
