# REPTIL - (pva) REPort uTILs


## Module for calculating/extracting specific memory usage and execution metrics from popvision reports


### Usage

An example usage might be the following:

```
import reptil

r = reptil.open_report("path/to/report.pop") # Open the pva report with reptil

r.memory.total() # Get the total memory (in Bytes) used on each tile.
```

You can also manipulate the pva report yourself:

```
import pva

pva_report = r.report # returns the underlying libpva report object
```
