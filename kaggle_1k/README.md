# KuaiRand-1K scale notebook

This isolated Kaggle kernel attaches `annanet/kuairand-1000`, the source
provided by the project owner. It validates the expected published 1K file
structure and writes an audit before benchmarking the actual catalog.

The notebook needs Internet enabled because it clones the immutable GitHub
commit at launch. It requests a Tesla P100, but `faiss-cpu` performs the index
operations on CPU; the output explicitly records the available FAISS GPU count
instead of claiming GPU index execution.

Launch after the attached source is available:

```sh
kaggle kernels push -p kaggle_1k
python3 scripts/watch_kaggle_kernel.py \
  --slug kushchaudhari/ranklab-kuairand-1k-scale \
  --output outputs/logs/kaggle_1k_scale.log
```
