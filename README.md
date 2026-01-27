# About

This is a Python based Paraview CGNS reader which support geometries that
change with each iteration (ex: moving objects).

It wraps the CGNS native library which is included in Paraview so it get good
performance. The wrapping is done with *ctypes* so there is no C/C++ to build.

It has been successfully used on Linux, Windows and MacOS with Paraview 5.9 to
6.0.

# Using

To use this reader in Paraview load `pvpycgnsreader.py` as a plugin using the
`Plugin Manager` from `Tools > Plugin Manager`. Then choose `Python-based CGNS
Reader` instead of `CGNS Series Reader`.

## Using without Paraview

It can also be used as a VTK reader:

```python
from pycgnsreader import PythonCNGSReader
reader = PythonCNGSReader()
reader.SetFileName(fname)
timesteps = reader.GetTimestepValues()
# Read the 4th time step
reader.SetTimeStep(timesteps[4])
reader.Update()
# a vtkMultiBlockDataSet
m = reader.GetOutputDataObject(0)
```

# Hacking

See:
* <https://docs.paraview.org/en/latest/ReferenceManual/pythonProgrammableFilter.html#python-algorithm>
* <https://cgns.github.io/CGNS_docs_current/cgio/>
