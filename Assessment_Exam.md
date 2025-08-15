# Programming Assessment Activity

These set of problems are designed to tackle the different programming concepts that are relevant for CSCI 214.

**Topics Covered**
- Data Conversion
- Object Oriented Programming
- Data Wrangling with Pandas

**Tools Needed**
- Python 3.x
- Your favorite IDE (i.e. VS Code)
- `torch` library (install via `pip install torch`)
- Cheat Sheet for OOP Python

---

## Data Conversion

### Example Problems

#### Data Conversion 1: List / NumPy → Tensor (reshape + normalization)

Given a Python list of pixel values `[0..255]` of length `12`, convert it to a `torch.float32` tensor shaped `(3, 2, 2)` and normalize to `[0, 1]`. Then also convert a NumPy array to a tensor without copying.

**Solution**

```python
import numpy as np
import torch

# From Python list -> tensor -> reshape -> normalize
pixels = [0, 64, 128, 255, 10, 20, 30, 40, 50, 60, 70, 80]
t = torch.tensor(pixels, dtype=torch.float32)        # create float tensor
t = t.view(3, 2, 2)                                  # reshape to (C,H,W) = (3,2,2)
t_norm = t / 255.0                                   # normalize to [0,1]

print(t_norm.shape, t_norm.dtype, t_norm.min().item(), t_norm.max().item())

# From NumPy -> tensor without copy (shares memory)
np_arr = np.arange(6, dtype=np.float32).reshape(2,3)
t_from_np = torch.from_numpy(np_arr)                 # zero-copy when dtypes compatible
np_arr[0, 0] = 999                                   # prove shared memory
print(t_from_np)                                     # first element reflects 999
```

#### Data Conversion 2: Integer labels ↔ one-hot tensor

Given class labels `labels = [2, 0, 1, 2]` with `num_classes = 3`:

1. Convert to a one-hot tensor of shape `(4, 3)` and dtype `torch.float32`.
2. Convert the one-hot tensor back to class indices.

**Solution**

```python
import torch
import torch.nn.functional as F

labels = torch.tensor([2, 0, 1, 2], dtype=torch.long)
num_classes = 3

# indices -> one-hot (long -> float32)
one_hot = F.one_hot(labels, num_classes=num_classes).to(torch.float32)
print(one_hot)
print(one_hot.shape, one_hot.dtype)  # (4, 3) float32

# one-hot -> indices
recovered = one_hot.argmax(dim=1)
print(recovered)                     # tensor([2, 0, 1, 2])
assert torch.equal(recovered, labels)
```

#### Data Conversion 3: Dtype & device conversions + Tensor ↔ NumPy

Start with a fake 8-bit image tensor on CPU shaped `(1, 3, 4, 4)`:

1. Convert uint8 → float32 in range `[0, 1]`.
2. Move to GPU if available and to float16 (half).
3. Bring back to CPU as float32 and convert to a NumPy arra

**Solution**

```python
import torch

# 1) uint8 -> float32 in [0,1]
img_u8 = torch.randint(0, 256, (1, 3, 4, 4), dtype=torch.uint8)  # fake image
img_f32 = img_u8.to(torch.float32) / 255.0
print(img_u8.dtype, "->", img_f32.dtype, img_f32.min().item(), img_f32.max().item())

# 2) CPU -> (optional) GPU and cast to float16
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
img_half = img_f32.to(device=device, dtype=torch.float16)
print("Device:", img_half.device, "dtype:", img_half.dtype)

# 3) Back to CPU float32 and to NumPy
img_back = img_half.to("cpu", dtype=torch.float32)
np_img = img_back.numpy()   # shares memory with CPU tensor
print(type(np_img), np_img.shape, np_img.dtype)
```

### Exercises

#### Exercise 1: Flatten, type change, and range scaling

You have a tensor of shape (2, 3, 4) containing torch.int32 values in the range [-100, 100].

1. Convert it to torch.float32.
2. Scale all values to the range [0, 1].
3. Flatten it into a 1D tensor.

#### Exercise 2: Indices ↔ One-hot ↔ Probabilities

You have a batch of 5 integer labels with possible values `0, 1, 2, 3`.

1. Convert them to a one-hot encoding.
2. Convert the one-hot tensor into probabilities by applying softmax along the class dimension.
3. Recover the original integer labels from the probabilities.

#### Exercise 3: CPU/GPU device and dtype conversions

You have a tensor of shape (10, 10) of random floats (float64) created on the CPU.

1. Move it to the GPU (if available) and convert it to float16.
2. Multiply it by 2.5 while still on GPU.
3. Bring it back to CPU as float32.


---

## Object Oriented Programming

### OOP 1: Shape Hierarchy

Store and compute areas of shapes using `torch` tensors.

**Classes**

- `Shape`: base class with `area()` (abstract)
- `Square(Shape)`: stores side length as a tensor, overrides `area()`
- `Circle(Shape)`: stores radius as a tensor, overrides `area()`

**Starter Code**

```python
import torch

class Shape:
    def area(self):
        raise NotImplementedError("Subclasses must implement area")

class Square(Shape):
    def __init__(self, side_length):
        # TODO: Create a tensor called self.side with data type torch.float32
        pass
    def area(self):
        # TODO: Return the square of self.side
        pass

class Circle(Shape):
    def __init__(self, radius):
        # TODO: Create a tensor called self.radius with data type torch.float32
        pass
    def area(self):
        # TODO: Return torch.pi times square of self.radius
        pass

# Example
sq = Square(4)
ci = Circle(3)
print("Square area:", sq.area().item())
print("Circle area:", ci.area().item())

```

### OOP 2: Simple Neural Layers

Demonstrate inheritance by making a base layer and two specific layers.

**Classes**

- `BaseLayer(nn.Module)`: base class with forward (abstract)
- `DoubleLayer(BaseLayer)`: multiplies input tensor by 2
- `AddBiasLayer(BaseLayer)`: adds a learnable bias

**Starter Code**

```python
import torch
import torch.nn as nn

class BaseLayer(nn.Module):
    def forward(self, x):
        raise NotImplementedError

class DoubleLayer(BaseLayer):
    # TODO: Return twice of x
    def forward(self, x):
        pass

class AddBiasLayer(BaseLayer):
    def __init__(self, size):
        super().__init__()
        # TODO: Create self.bias which is an nn.Parameter filled with zeros using torch.zeros with size as its dimensionality
        pass
    def forward(self, x):
        # TODO: Return x plus the bias
        pass

# Example
x = torch.tensor([1.0, 2.0, 3.0])
dl = DoubleLayer()
abl = AddBiasLayer(3)
print("Double:", dl(x))
print("AddBias:", abl(x))
```

### OOP 3: Dataset Variants

Create a base dataset and two simple derived datasets.

**Classes**

- `BaseDataset(torch.utils.data.Dataset)`: stores data & labels
- `SquaredDataset(BaseDataset)`: returns squared data
- `NormalizedDataset(BaseDataset)`: returns normalized data

**Starter Code**

```python
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data, labels):
        self.data = # TODO: Create a tensor self.data from data with data type torch.float32
        self.labels = # TODO: Create a tensor self.labels from labels
    def __len__(self):
        # TODO: Return the length of self.data
        pass
    def __getitem__(self, idx):
        # TODO: Return an x, y representation from self.data using idx
        pass

class SquaredDataset(BaseDataset):
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x ** 2, y

class NormalizedDataset(BaseDataset):
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return (x - x.mean()) / x.std(), y

# Example
data = [[1,2],[3,4],[5,6]]
labels = [0,1,0]
sq_ds = SquaredDataset(data, labels)
norm_ds = NormalizedDataset(data, labels)
print("Squared first item:", sq_ds[0])
print("Normalized first item:", norm_ds[0])
```


### OOP 4: Metric Calculation Framework

Create a base metric class and two specific metrics for evaluation.

**Classes**

- `BaseMetric`: stores running totals, defines reset, update, and compute
- `AccuracyMetric(BaseMetric)`: tracks classification accuracy
- `MAEMetric(BaseMetric)`: tracks mean absolute error

**Starter Code**

```python
import torch

class BaseMetric:
    def reset(self):
        raise NotImplementedError
    def update(self, preds, targets):
        raise NotImplementedError
    def compute(self):
        raise NotImplementedError

class AccuracyMetric(BaseMetric):
    def reset(self):
        self.correct = 0
        self.total = 0
    def update(self, preds, targets):
        preds = preds.argmax(dim=1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.numel()
    def compute(self):
        # TODO: Return the result of accuracy as number of correct over total. If total is 0 or less, return 0
        pass

# TODO: Research on MAE and implement update and compute
class MAEMetric(BaseMetric):
    def reset(self):
        self.error_sum = 0.0
        self.total = 0
    def update(self, preds, targets):
        pass
    def compute(self):
        pass

# Example
acc = AccuracyMetric()
mae = MAEMetric()

preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
targets_cls = torch.tensor([1, 0])
targets_reg = torch.tensor([0.0, 1.0])

acc.reset(); acc.update(preds, targets_cls)
mae.reset(); mae.update(preds[:,0], targets_reg)

print("Accuracy:", acc.compute())
print("MAE:", mae.compute())
```

---

## Data Wrangling with Pandas

### Pandas + OOP 1: CSV Loader + Cleaner + Aggregator

Load a CSV, clean missing data, and compute aggregates via an OOP pipeline.

**Classes**

- `CSVLoader`: loads a CSV into a DataFrame
- `DataCleaner`: fills missing values and drops duplicates
- `Aggregator`: groups by a column and computes statistics

**Starter Code**

```python
import pandas as pd

class CSVLoader:
    def __init__(self, path):
        self.path = path
    def load(self):
        return pd.read_csv(self.path)

class DataCleaner:
    def __init__(self, fill_value=0):
        self.fill_value = fill_value
    def clean(self, df):
        # TODO: Returne a df that fills na values with self.fill_value and drop duplicates
        return df

class Aggregator:
    def __init__(self, group_col):
        self.group_col = group_col
    def aggregate(self, df):
        return df.groupby(self.group_col).mean(numeric_only=True)

# Example
# loader = CSVLoader("data.csv")
# cleaner = DataCleaner(fill_value=0)
# agg = Aggregator("category")
# df = loader.load()
# df = cleaner.clean(df)
# print(agg.aggregate(df))
```

### Pandas + OOP 2: Column Transformer Framework

Create a base transformer for column operations and two specific column transformers.

**Classes**

- `BaseColumnTransformer`: defines interface for transforming a DataFrame column
- `NormalizeColumn`: scales a numeric column to [0,1] range
- `EncodeCategory`: label-encodes a categorical column

**Starter Code**

```python
import pandas as pd

class BaseColumnTransformer:
    def __init__(self, col):
        self.col = col
    def transform(self, df):
        raise NotImplementedError

class NormalizeColumn(BaseColumnTransformer):
    # TODO: Create a normalized version of the dataframe using (x - min) / (max - min)
    def transform(self, df):
        return df

class EncodeCategory(BaseColumnTransformer):
    def transform(self, df):
        df[self.col] = df[self.col].astype("category").cat.codes
        return df

# Example
# df = pd.DataFrame({"val":[10,20,30], "cat":["A","B","A"]})
# norm = NormalizeColumn("val")
# enc = EncodeCategory("cat")
# df = norm.transform(df)
# df = enc.transform(df)
# print(df)
```

### Pandas + OOP 3: Merging and Filtering DataFrames

Manage two DataFrames, merge them, and filter results.

**Classes**

- `DataFrameManager`: holds two DataFrames
- `Merger`: merges them on a key column
- `Filter`: filters merged DataFrame based on a condition

**Starter Code**

```python
import pandas as pd

class DataFrameManager:
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2

class Merger:
    def __init__(self, key):
        self.key = key
    def merge(self, manager):
        # TODO: Return the merged version of self.df1 and self.df2 based on self.key
        pass

class Filter:
    def __init__(self, col, threshold):
        self.col = col
        self.threshold = threshold
    def filter(self, df):
        # TODO: Return a df filtered where self.col > self.threshold
        return df

# Example
# df1 = pd.DataFrame({"id":[1,2,3],"val1":[10,20,30]})
# df2 = pd.DataFrame({"id":[1,2,3],"val2":[5,25,35]})
# mgr = DataFrameManager(df1, df2)
# merger = Merger("id")
# flt = Filter("val2", 20)
# merged = merger.merge(mgr)
# print(flt.filter(merged))
```
