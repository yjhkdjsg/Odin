# Python Libraries API & Version Migration Guide (Python 3.12+, Pandas 2.x/3.x, NumPy 2.x, Scikit-Learn 1.4+)

## Pandas 2.x and 3.x Breaking Changes

- `DataFrame.append()` and `Series.append()` were removed in pandas 2.0. Always use `pd.concat([df1, df2], ignore_index=True)`. To append a dict row, wrap it as `pd.DataFrame([row_dict])` inside `pd.concat`.
- `DataFrame.iteritems()` was removed in pandas 2.0. Use `df.items()` instead.
- `DataFrame.applymap()` was deprecated in pandas 2.1 and removed in pandas 3.0. Use `df.map(func)` instead.
- `DataFrame.fillna(method="ffill")` and `fillna(method="bfill")` were removed in pandas 3.0. Call `df.ffill()` or `df.bfill()` directly instead.
- `DataFrame.groupby(...).sum()` in pandas 2.0+ defaults to `numeric_only=False`; pass `numeric_only=True` when DataFrames include string columns.
- `DataFrame.swapaxes()` was removed in pandas 3.0; use `.T` or `.transpose()`.
- When creating named aggregations with `groupby().agg()`, pass keyword tuples: `df.groupby("k", as_index=False).agg(total=("v", "sum"), mean_v=("v", "mean"))`.

## NumPy 2.x Breaking Changes

- Legacy aliases `np.int`, `np.float`, `np.bool`, `np.complex`, `np.object`, `np.str` were removed in NumPy 1.24+. Use Python built-ins (`int`, `float`, `bool`) or explicit NumPy dtypes (`np.int64`, `np.float64`, `np.bool_`).
- `np.trapz` was deprecated in NumPy 2.0 and removed in NumPy 2.4+. Use `np.trapezoid(y, x=x)` instead.
- `np.in1d` was removed in NumPy 2.x. Use `np.isin(ar1, ar2)` instead.
- `np.row_stack` was removed in favor of `np.vstack`.
- `np.matrix` is deprecated; use standard 2D `np.ndarray` (`np.array(...)`) and the `@` operator or `np.matmul(a, b)` for matrix multiplication.
- `np.ptp(a)` works as a function, while `a.ptp()` method was removed in NumPy 2.0. Always use `np.ptp(arr)`.
- `np.product` and `np.cumproduct` were removed in NumPy 2.0; use `np.prod` and `np.cumprod`.

## Scikit-Learn 1.4+ API Reference

- `OneHotEncoder(sparse=False)` raises `TypeError: got an unexpected keyword argument 'sparse'`. Always use `OneHotEncoder(sparse_output=False)`.
- `mean_squared_error(..., squared=False)` was removed in scikit-learn 1.6+; use `root_mean_squared_error(y_true, y_pred)` instead.
- `sklearn.datasets.load_boston` was removed; use `fetch_california_housing` or synthetic `make_regression`.

## Pydantic v2 API Reference

- Replace `.dict()` with `.model_dump()` and `.json()` with `.model_dump_json()`.
- Replace `.parse_obj(data)` with `.model_validate(data)` and `.parse_raw(s)` with `.model_validate_json(s)`.
- Replace `@validator("field")` with `@field_validator("field")` (decorated with `@classmethod`).
- Replace inner `class Config:` with `model_config = ConfigDict(...)`.

## Standard Library Uncommon Patterns (Python 3.12+)

- `itertools.groupby(iterable, key=None)` only groups *consecutive* elements with the same key. Sort the data by `key` before passing it to `groupby`. Note that each group iterator must be consumed before advancing to the next group.
- `functools.reduce(fn, iterable, initializer)` raises `TypeError` on empty iterables unless `initializer` is provided.
- `functools.lru_cache` requires all arguments to be hashable (convert lists to tuples).
- `heapq` implements a min-heap. For a max-heap, negate numeric priorities; when tuples `(priority, obj)` can tie on priority and `obj` is not comparable (e.g. dicts or custom objects), insert a unique counter `(priority, count, obj)`.
- `asyncio.run(coro)` creates a brand-new event loop and fails with `RuntimeError` if called inside an already-running loop; inside async code use `await`, `asyncio.create_task()`, or `asyncio.gather()`.
- `collections.abc.MutableMapping` requires implementing five abstract methods: `__getitem__`, `__setitem__`, `__delitem__`, `__iter__`, and `__len__`.
- `datetime.datetime.utcnow()` is deprecated in Python 3.12+; use `datetime.datetime.now(datetime.timezone.utc)`.
