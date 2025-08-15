# Python OOP Cheat Sheet

## Basic Class

```python
class MyClass:
    def __init__(self, name):
        self.name = name  # attribute

    def greet(self):
        return f"Hello, {self.name}"

obj = MyClass("Alice")
print(obj.greet())  # Hello, Alice
```

## Inheritance and Overriding

```python
class Animal:
    def speak(self):
        return "Some sound"

class Dog(Animal):
    def speak(self):  # override
        return "Woof"

a = Animal()
d = Dog()
print(a.speak(), d.speak())  # Some sound Woof
```

## `super()` (Call Parent)

```python
class Parent:
    def __init__(self, value):
        self.value = value

class Child(Parent):
    def __init__(self, value, extra):
        super().__init__(value)  # call parent __init__
        self.extra = extra
```

## Special Methods

| Method        | Purpose                     |
| ------------- | --------------------------- |
| `__len__`     | Length of object            |
| `__getitem__` | Index access                |
| `__call__`    | Make object callable        |
| `__repr__`    | Debug string representation |

```python
class MyList:
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def __repr__(self):
        return f"MyList({self.data})"
```

## PyTorch `nn.Module`

```python
import torch
import torch.nn as nn

class MyLayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size, size))  # trainable
        self.register_buffer("scale", torch.tensor(1.0))     # non-trainable

    def forward(self, x):
        return x @ self.weight * self.scale

layer = MyLayer(3)
x = torch.randn(2, 3)
print(layer(x).shape)
```

## PyTorch `Dataset`

```python
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SquaredDataset(BaseDataset):
    def __getitem__(self, idx):  # override
        x, y = super().__getitem__(idx)
        return x ** 2, y
```
