# ODIN Knowledge Base - Error-Fix Examples

This file contains curated error-fix examples for the ODIN Adaptive RAG system.
Format: Each example has a category, error, and fix.

# Example: Pandas DataFrame append deprecated and removed
Category: version_specific
Error:
```
AttributeError: 'DataFrame' object has no attribute 'append'
```
Fix:
```python
import pandas as pd
# DataFrame.append() was removed in pandas 2.0+.
# Use pd.concat() instead:
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
```

# Example: Pandas DataFrame iteritems removed
Category: version_specific
Error:
```
AttributeError: 'DataFrame' object has no attribute 'iteritems'
```
Fix:
```python
# DataFrame.iteritems() was removed in pandas 2.0+.
# Use df.items() instead:
for col_name, series in df.items():
    print(col_name, series.tolist())
```

# Example: Pandas DataFrame applymap removed in pandas 3
Category: version_specific
Error:
```
AttributeError: 'DataFrame' object has no attribute 'applymap'
```
Fix:
```python
# DataFrame.applymap() was deprecated in pandas 2.1 and removed in pandas 3.0.
# Use DataFrame.map() instead:
df = df.map(lambda x: x * 2)
```

# Example: Pandas fillna method argument removed
Category: version_specific
Error:
```
TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'
```
Fix:
```python
# fillna(method='ffill') / fillna(method='bfill') was deprecated in pandas 2.1 and removed in pandas 3.0.
# Use df.ffill() or df.bfill() directly:
df_filled = df.ffill()
```

# Example: NumPy np.int and np.float removed
Category: version_specific
Error:
```
AttributeError: module 'numpy' has no attribute 'int'
```
Fix:
```python
import numpy as np
# Aliases np.int, np.float, np.bool were removed in NumPy 1.24+.
# Use built-in int/float/bool or explicit np.int64 / np.float64:
arr = np.array([1, 2, 3], dtype=np.int64)
```

# Example: NumPy trapz removed in NumPy 2
Category: version_specific
Error:
```
AttributeError: module 'numpy' has no attribute 'trapz'
```
Fix:
```python
import numpy as np
# np.trapz was deprecated in NumPy 2.0 and removed in NumPy 2.4+.
# Use np.trapezoid(y, x) instead:
area = np.trapezoid(y, x)
```

# Example: NumPy in1d removed in NumPy 2
Category: version_specific
Error:
```
AttributeError: module 'numpy' has no attribute 'in1d'
```
Fix:
```python
import numpy as np
# np.in1d was deprecated and removed in NumPy 2.x.
# Use np.isin(element, test_elements) instead:
mask = np.isin(arr, targets)
```

# Example: NumPy matrix deprecated
Category: version_specific
Error:
```
PendingDeprecationWarning: the matrix subclass is not the recommended way
```
Fix:
```python
import numpy as np
# Use np.array and @ operator for matrix multiplication instead of np.matrix:
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])
product = m1 @ m2
```

# Example: Scikit-learn OneHotEncoder sparse parameter removed
Category: version_specific
Error:
```
TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'
```
Fix:
```python
from sklearn.preprocessing import OneHotEncoder
# The `sparse` parameter was removed in scikit-learn 1.4+.
# Use `sparse_output=False` instead:
enc = OneHotEncoder(sparse_output=False)
```

# Example: Pydantic v2 model_dump replaces dict and field_validator replaces validator
Category: version_specific
Error:
```
AttributeError: 'MyModel' object has no attribute 'parse_obj'
```
Fix:
```python
from pydantic import BaseModel, field_validator
# In Pydantic v2, use model_validate() instead of parse_obj(),
# model_dump() instead of dict(), and @field_validator instead of @validator.
```

# Example: Requests timeout not set
Category: library_api
Error:
```
The request hangs indefinitely without returning
```
Fix:
```python
import requests
# Always pass timeout=(connect, read) or timeout=seconds:
response = requests.get(url, timeout=10)
response.raise_for_status()
```

# Example: Pandas read_csv wrong separator
Category: library_api
Error:
```
ParserError: Error tokenizing data. C error: Expected 3 fields, saw 5
```
Fix:
```python
import pandas as pd
# Specify the exact delimiter with sep= (e.g., '\t' or ';'):
df = pd.read_csv('file.csv', sep=';')
```

# Example: Pandas groupby agg named aggregation syntax
Category: library_api
Error:
```
TypeError: DataFrameGroupBy.agg() got an unexpected keyword argument
```
Fix:
```python
import pandas as pd
# Use NamedAgg tuples (column_name, aggfunc) with groupby().agg():
summary = df.groupby('group', as_index=False).agg(
    total=('value', 'sum'),
    avg=('value', 'mean'),
)
```

# Example: Dictionary KeyError vs get and defaultdict
Category: library_api
Error:
```
KeyError: 'missing_key'
```
Fix:
```python
from collections import defaultdict
# Use dict.get(key, default) or collections.defaultdict(int) for safe lookup:
counts = defaultdict(int)
counts['missing_key'] += 1
```

# Example: JSON serialization of NumPy scalar or set
Category: library_api
Error:
```
TypeError: Object of type int64 is not JSON serializable
```
Fix:
```python
import json
# Convert NumPy scalars using .item() or sets using list() before json.dumps:
payload = {"val": int(np_val), "items": sorted(list(my_set))}
text = json.dumps(payload)
```

# Example: itertools groupby requires sorted data
Category: uncommon_pattern
Error:
```
itertools.groupby produces unexpected groups
```
Fix:
```python
from itertools import groupby
# itertools.groupby only groups consecutive equal keys.
# Always sort by the same key function first:
data = ['apple', 'banana', 'apple', 'cherry']
counts = {k: len(list(g)) for k, g in groupby(sorted(data))}
```

# Example: functools.reduce with empty sequence
Category: uncommon_pattern
Error:
```
TypeError: reduce() of empty sequence with no initial value
```
Fix:
```python
from functools import reduce
# Always supply an initializer as the third argument when the iterable may be empty:
result = reduce(lambda x, y: x + y, numbers, 0)
```

# Example: asyncio run cannot be called from running event loop
Category: uncommon_pattern
Error:
```
RuntimeError: asyncio.run() cannot be called from a running event loop
```
Fix:
```python
import asyncio
# Inside an async function, await coroutines directly or use asyncio.create_task / asyncio.gather.
# Only call asyncio.run() from top-level synchronous code:
async def main():
    results = await asyncio.gather(coro_a(), coro_b())
    return results
```

# Example: heapq tie-breaking with uncomparable objects
Category: uncommon_pattern
Error:
```
TypeError: '<' not supported between instances of 'dict' and 'dict'
```
Fix:
```python
import heapq
# When pushing (priority, payload) tuples into heapq where priorities may tie,
# include a monotonically increasing counter as tie-breaker: (priority, idx, payload).
heap = []
for idx, (prio, obj) in enumerate(items):
    heapq.heappush(heap, (prio, idx, obj))
```

# Example: collections.abc MutableMapping requires five abstract methods
Category: uncommon_pattern
Error:
```
TypeError: Can't instantiate abstract class CustomDict without an implementation for abstract methods '__delitem__', '__iter__', '__len__'
```
Fix:
```python
from collections.abc import MutableMapping
# Subclassing MutableMapping requires implementing:
# __getitem__, __setitem__, __delitem__, __iter__, and __len__.
```
