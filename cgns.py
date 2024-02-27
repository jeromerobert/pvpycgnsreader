from collections import namedtuple
import ctypes
from ctypes import (
    POINTER,
    create_string_buffer,
    c_char_p,
    c_int,
    c_void_p,
    c_double,
    c_size_t,
)
import sys
import numpy as np

# Possible CGNS exception types
CGNSError = type("CGNSError", (Exception,), {})
CGNSNodeNotFound = type("CGNSNodeNotFound", (CGNSError,), {})
CGNSIncorrectPath = type("CGNSIncorrectPath", (CGNSError,), {})
CGNSNoIndexDim = type("CGNSNoIndexDim", (CGNSError,), {})


def _load_lib(root):
    if sys.platform == "darwin":
        templates = [
            "@executable_path/../Libraries/lib%s.dylib",
            "lib%s.dylib",
            "lib%s.so",
            "lib%s.bundle",
            "%s.dylib",
            "%s.so",
            "%s.bundle",
            "%s",
        ]
    elif sys.platform == "win32" or sys.platform == "cygwin":
        templates = ["%s.dll", "lib%s.dll"]
    else:
        templates = ["lib%s.so", "lib%s.so.4.4", "lib%s.so.4.2"]
    for t in templates:
        try:
            return ctypes.CDLL(t % root)
        except OSError:
            pass
    raise OSError("Cannot load the %s library" % root)


CG_MODE_READ = 0
CG_MODE_WRITE = 1
CG_MODE_MODIFY = 2
CG_MODE_CLOSED = 3

CG_FILE_NONE = 0
CG_FILE_ADF = 1
CG_FILE_HDF5 = 2
CG_FILE_ADF2 = 3

CGNSNode = namedtuple(
    "CGNSNode", ["name", "id", "children", "dtype", "dimensions", "label"]
)

_CGNS_TO_NUMPY = {
    "MT": None,
    "C1": "S",
    "I4": "i4",
    "I8": "i8",
    "U4": "u4",
    "U8": "u8",
    "R4": "f4",
    "R8": "f8",
    "B1": "i1",
}


# Possible return codes
# https://cgns.github.io/CGNS_docs_current/cgio/errors.html
CGNS_STATUS = {
    1: CGNSError,
    2: CGNSNodeNotFound,
    3: CGNSIncorrectPath,
    4: CGNSNoIndexDim,
    8: Exception("ADF file open error"),
    65: Exception("CHILDREN_IDS_NOT_FOUND"),
    33: Exception("Node had no data associated with it"),
    76: Exception("H5Gopen:open of a node group failed"),
    71: Exception("Node attribute doesn't exist"),
    85: Exception("H5Dread:read of node data failed"),
}

class _CGN:
    def __init__(self, name):
        self.name = name

def _errcheck(status, fn, arg):
    if status != 0:
        print(status, fn, arg)
        try:
            raise CGNS_STATUS[status]
        except KeyError:
            raise CGNSError


class _Prefixer:
    def __init__(self, delegate, prefix):
        self.delegate = delegate
        self.prefix = prefix

    def __getattr__(self, name):
        return getattr(self.delegate, self.prefix + name)


def _pvversion():
    """Return the paraview version string"""
    try:
        from paraview import servermanager
        maj = servermanager.vtkSMProxyManager.GetVersionMajor()
        mn = servermanager.vtkSMProxyManager.GetVersionMinor()
        return "{}.{}".format(maj, mn)
    except ModuleNotFoundError:
        return None


class _CGNSWrappers:
    def __init__(self):
        pv_version = _pvversion()
        if pv_version:
            # This is the name of the CGNS lib in Paraview
            libname, self.prefix, self.cgsize_t = "vtkcgns-pv"+_pvversion(), "vtkcgns_cgio_", c_size_t
        else:
            # System or conda CGNS
            libname, self.prefix, self.cgsize_t = "cgns", "cgio_", c_size_t
        self.lib = _load_lib(libname)
        # ier = cgio_open_file(const char *filename, int file_mode, int file_type, int *cgio_num);
        self._proto("open_file", [c_char_p, c_int, c_int, POINTER(c_int)])
        # ier = cgio_number_children(int cgio_num, double id, int *num_child);
        self._proto("number_children", [c_int, c_double, POINTER(c_int)])
        # ier = cgio_get_root_id(int cgio_num, double *rootid);
        self._proto("get_root_id", [c_int, POINTER(c_double)])
        # ier = cgio_children_ids(int cgio_num, double id, int start, int max_ret, int *num_ret, double *child_ids);
        self._proto(
            "children_ids",
            [c_int, c_double, c_int, c_int, POINTER(c_int), POINTER(c_double)],
        )
        # int cgio_get_name (int cgio_num, double id, char *name);
        self._proto("get_name", [c_int, c_double, c_char_p])
        # ier = cgio_get_data_type(int cgio_num, double id, char *data_type);
        self._proto("get_data_type", [c_int, c_double, c_char_p])
        # ier = cgio_get_dimensions(int cgio_num, double id, int *ndims, cgsize_t *dims);
        self._proto(
            "get_dimensions", [c_int, c_double, POINTER(c_int), POINTER(self.cgsize_t)]
        )
        # ier = cgio_get_label(int cgio_num, double id, char *label);
        self._proto("get_label", [c_int, c_double, c_char_p])
        # ier = cgio_read_all_data_type(int cgio_num, double id, const char *m_data_type, void *data);
        self._proto("read_all_data_type", [c_int, c_double, c_char_p, c_void_p])
        self.lib = _Prefixer(self.lib, self.prefix)

    def _proto(self, fname, argtypes):
        r = getattr(self.lib, self.prefix + fname)
        r.argtypes = argtypes
        r.errcheck = _errcheck

    def open(self, name):
        file = c_int()
        name = bytes(name, "utf-8")
        self.lib.open_file(name, CG_MODE_READ, CG_FILE_NONE, ctypes.byref(file))
        rootid = c_double()
        self.lib.get_root_id(file, ctypes.byref(rootid))
        return file, self._create_node(file, rootid)

    def _create_node(self, cgio_num, node_id):
        nc = c_int()
        self.lib.number_children(cgio_num, node_id, ctypes.byref(nc))
        nc = nc.value
        buf = create_string_buffer(32)
        self.lib.get_name(cgio_num, node_id, buf)
        name = buf.value.decode("utf-8")
        self.lib.get_label(cgio_num, node_id, buf)
        label = buf.value.decode("utf-8")
        self.lib.get_data_type(cgio_num, node_id, buf)
        dtype = buf.value.decode("utf-8")
        ndim = c_int()
        dims = (self.cgsize_t * 12)()
        self.lib.get_dimensions(cgio_num, node_id, ctypes.byref(ndim), dims)
        dims = list(dims[: ndim.value])
        children = {}
        if nc > 0:
            ids = (c_double * nc)()
            num_ret = c_int()
            self.lib.children_ids(cgio_num, node_id, 1, nc, ctypes.byref(num_ret), ids)
            for i in ids:
                c = self._create_node(cgio_num, c_double(i))
                children[c.name] = c
        return CGNSNode(
            name=name,
            id=node_id,
            children=children,
            dtype=dtype,
            dimensions=dims,
            label=label,
        )

    def read_data(self, cgio_num, node):
        dim = list(reversed(node.dimensions))
        isstring = node.dtype == "C1"
        if isstring:
            dtype = "|S{}".format(dim[-1])
            dim = dim[:-1]
        else:
            dtype = _CGNS_TO_NUMPY[node.dtype]
        buf = np.zeros(dim, dtype=dtype)
        self.lib.read_all_data_type(
            cgio_num, node.id, node.dtype.encode(), buf.ctypes.data
        )
        if isstring:
            if len(buf.shape) > 0:
                buf = buf.tolist()
                _bytetostr(buf)
            else:
                buf = buf.tobytes().decode("utf-8").strip()
        return buf


_cgns_wrapper = _CGNSWrappers()


def _bytetostr(alist):
    """Convert a list of numpy array of char to a list of string"""
    for i, e in enumerate(alist):
        if isinstance(e, list):
            _bytetostr(e)
        else:
            # This should not happen in valid CGNS file
            if len(alist[i]) == 0 or alist[i][0] == 0:
                alist[i] = None
                continue
            try:
                alist[i] = e.decode("utf-8").strip()
            except UnicodeDecodeError:
                alist[i] = None
            if alist[i] == "Null":
                alist[i] = None


def _get_path(cgnsnode, path):
    r = cgnsnode.children.get(path[0])
    if r is None or len(path) == 1:
        return r
    else:
        return _get_path(r, path[1:])


def _by_labels(cgnsnode, labels):
    r = child_with_label(cgnsnode, labels[0])
    if len(r) == 0:
        return []
    elif len(labels) == 1:
        return r
    else:
        l = []
        for c in r:
            l.extend(_by_labels(c, labels[1:]))
        return l


def child_with_label(node, label):
    r = []
    for c in node.children.values():
        if c.label == label:
            r.append(c)
    return r


def child_with_name(node, name):
    if node is None:
        return None
    if isinstance(node, list):
        node = node[0]
    return node.children.get(name)

def find_node(n, args):
    if len(args) == 0:
        return n
    else :
        if type(args[0]) == _CGN :
            n = child_with_name(n, args[0].name)
        else :
            n = child_with_label(n, args[0])
        args = args[1:]
        return find_node(n, args)



class Reader:
    def __init__(self, filename):
        # FIXME add a close method
        self.cgio_num, self.data = _cgns_wrapper.open(filename)

    def nodes_by_labels(self, labels=[]):
        return _by_labels(self.data, labels)

    def data_by_labels(self, labels=[]):
        n = self.nodes_by_labels(labels)
        return None if len(n) == 0 else self.read_array(n[0])

    def node(self, path=[]):
        """Return the meta data of a node"""
        # FIXME Using names to navigates in a CGNS files is wrong. I should use
        # the label. This needs to modify the whole logic for the pvpycgnsreader
        # module.
        return _get_path(self.data, path)

    def read_array(self, node):
        if node is None:
            return None
        return _cgns_wrapper.read_data(self.cgio_num, node)

    def read_path(self, path):
        """Read the data associated to a path"""
        n = self.node(path)
        return None if n is None else self.read_array(n)


if __name__ == "__main__":
    r = Reader("test.cgns")
    print(r.read_path(["Base", "TimeIterValues", "TimeValues"]))
    print(r.read_path(["Base", "0000001_T#1_WALL_2_1", "Elem"]))
    print(r.node(["Base", "TimeIterValues", "ZonePointers"]))
    print(r.read_path(["Base", "TimeIterValues", "ZonePointers"]))
