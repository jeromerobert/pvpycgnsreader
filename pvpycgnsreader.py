
from paraview.util.vtkAlgorithm import *
import sys
import os

sys.path.append(os.path.dirname(__file__))
# Flush the cgns module when paraview reload the module
sys.modules.pop("cngs", None)
import cgns
import numpy as np
from cgns import _CGN


def createModifiedCallback(anobject):
    import weakref

    weakref_obj = weakref.ref(anobject)
    anobject = None

    def _markmodified(*args, **kwars):
        o = weakref_obj()
        if o is not None:
            o.Modified()

    return _markmodified


# Map CGNS type to (VTK type, VTK num vertex)
# according to IO/CGNS/vtkCGNSReaderInternal.cxx and Common/DataModel/vtkCellType.h
CELL_TYPE = {
    7: (9, 4),  # QUAD_4 / VTK_QUAD
    5: (5, 3),  # TRI_3 / VTK_TRIANGLE
    17: (12, 8),  # HEXA_8 / VTK_HEXAHEDRON
    12: (14, 5), # PYRA_5 / VTK_PYRAMID
    14: (13, 6), #PENTA_6 / VTK_WEDGE
    10: (10, 4) # TETRA_4 / VTK_TETRA
}


def _numpy_to_cell_array(cell_types, offset, connectivity):
    """
    Create a vtkCellArray from 2 numpy arrays and a vtkUnsignedCharArray cell type
    array from a numpy array
    """
    from vtkmodules.vtkCommonDataModel import vtkCellArray
    from vtkmodules.util.vtkConstants import VTK_ID_TYPE, VTK_UNSIGNED_CHAR
    from vtk.util.numpy_support import numpy_to_vtk
    ca = vtkCellArray()
    ca.SetData(
        numpy_to_vtk(offset, deep=1, array_type=VTK_ID_TYPE),
        numpy_to_vtk(connectivity, deep=1, array_type=VTK_ID_TYPE),
    )
    ct = numpy_to_vtk(cell_types, deep=1, array_type=VTK_UNSIGNED_CHAR)
    return ct, ca


@smproxy.reader(
    name="PythonCGNSReader",
    label="Python-based CGNS Reader",
    extensions="cgns",
    file_description="CGNS files",
)
class PythonCNGSReader(VTKPythonAlgorithmBase):
    """A reader that reads a CSV file. If the CSV has a "time" column, then
    the data is treated as a temporal dataset"""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=1, outputType="vtkMultiBlockDataSet"
        )
        self._filename = None
        from vtkmodules.vtkCommonCore import vtkDataArraySelection

        self._arrayselection = vtkDataArraySelection()
        self._arrayselection.AddObserver("ModifiedEvent", createModifiedCallback(self))
        self._modified = False
        self._reader = None
        self._np_arrays = []

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions="cgns", file_description="CGNS files")
    def SetFileName(self, name):
        """Specify filename for the file to read."""
        if self._filename != name:
            self._filename = name
            self.Modified()
            self._modified = True

    def _get_timesteps(self):
        if self._filename is None:
            return []
        if self._modified or self._reader is None:
            self._reader = cgns.Reader(self._filename)
            self._modified = False
        base_iter = self._reader.nodes_by_labels(["CGNSBase_t", "BaseIterativeData_t"])
        if base_iter == []:
            return None
        base_iter = base_iter[0]
        time_node = cgns.child_with_name(base_iter, "TimeValues")
        if time_node is None:
            time_node = cgns.child_with_name(base_iter, "IterationValues")
        if time_node is None:
            return None
        return self._reader.read_array(time_node)

    def _get_update_time(self, outInfo):
        executive = self.GetExecutive()
        timesteps = self._get_timesteps()
        if timesteps is None or len(timesteps) == 0:
            return None
        elif outInfo.Has(executive.UPDATE_TIME_STEP()) and len(timesteps) > 0:
            utime = outInfo.Get(executive.UPDATE_TIME_STEP())
            dtime = timesteps[0]
            for atime in timesteps:
                if atime > utime:
                    return dtime
                else:
                    dtime = atime
            return dtime
        else:
            assert len(timesteps) > 0
            return timesteps[0]

    @smproperty.doublevector(
        name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty"
    )
    def GetTimestepValues(self):
        return self._get_timesteps()

    # Array selection API is typical with readers in VTK
    # This is intended to allow ability for users to choose which arrays to
    # load. To expose that in ParaView, simply use the
    # smproperty.dataarrayselection().
    # This method **must** return a `vtkDataArraySelection` instance.
    @smproperty.dataarrayselection(name="Arrays")
    def GetDataArraySelection(self):
        return self._arrayselection

    def RequestInformation(self, request, inInfoVec, outInfoVec):
        timesteps = self._get_timesteps()
        executive = self.GetExecutive()
        outInfo = outInfoVec.GetInformationObject(0)
        if timesteps is not None:
            outInfo.Set(executive.TIME_STEPS(), timesteps, len(timesteps))
            outInfo.Set(executive.TIME_RANGE(), timesteps[[0, -1]], 2)
        return 1

    def _read_grid_coordinates(self, zone):
        # FIXME: allocate the return array and fill it to avoid a memory peak
        r = []
        base = self._reader.nodes_by_labels(["CGNSBase_t"])[0]
        for l in ["X", "Y", "Z"]:
            coord = cgns.find_node(base, [_CGN(zone), "GridCoordinates_t", _CGN("Coordinate" + l)])
            a = self._reader.read_array(coord)
            r.append(a.astype(np.single).reshape(-1, 1))
        return np.hstack(r)

    def _create_ug_mixed(self, elem_nodes):
        T = []
        O = []
        C = []
        nb_elem_nodes = len(elem_nodes)
        # FIXME iterate over nodes in the "ElementRange" order
        for i in range(nb_elem_nodes):
            elem_name = elem_nodes[i].name
            c = cgns.child_with_name(elem_nodes[i], "ElementConnectivity")
            c_offset = cgns.child_with_name(elem_nodes[i], "ElementStartOffset")
            offsets_cgns = self._reader.read_array(c_offset)
            num_cells = len(offsets_cgns)-1
            cells_cgns = self._reader.read_array(c)
            types_elem = cells_cgns[offsets_cgns[:-1]]
            types_elem_vtk = np.zeros(num_cells, dtype=np.ubyte)
            cells_sizes = np.zeros(num_cells, dtype=np.int)
            for cgns_type, (vtk_ctype, vtk_csize) in CELL_TYPE.items():
                elem = types_elem == cgns_type
                types_elem_vtk[elem] = vtk_ctype
                cells_sizes[elem] = vtk_csize
            mask = np.ones_like(cells_cgns, dtype=bool)
            mask[offsets_cgns[:-1]] = False
            cells_vtk = cells_cgns[mask] - 1
            T.append(types_elem_vtk)
            O.append(cells_sizes)
            C.append(cells_vtk)
        return np.concatenate(T), np.concatenate(O), np.concatenate(C)

    def _create_ug_notmixed(self, element_node, celltype):
        c = cgns.child_with_name(element_node, "ElementConnectivity")
        cells = self._reader.read_array(c) - 1
        cellsize = CELL_TYPE[celltype][1]
        ncells = cells.shape[0] // cellsize
        celltypes = np.full((ncells,), CELL_TYPE[celltype][0], dtype=np.ubyte)
        cellsizes = np.full((ncells,), cellsize, dtype=np.int)
        return celltypes, cellsizes, cells

    def _create_unstructured_grid(self, zone):
        from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
        from vtkmodules.numpy_interface import dataset_adapter as dsa

        ug = vtkUnstructuredGrid()
        pug = dsa.WrapDataObject(ug)
        coords = self._read_grid_coordinates(zone)
        self._np_arrays.append(coords)
        pug.SetPoints(coords)
        base = self._reader.nodes_by_labels(["CGNSBase_t"])[0]
        elem_nodes = cgns.find_node(base, [_CGN(zone), "Elements_t"])
        if len(elem_nodes) == 0:
            print("WARNING: No elements founds")
            return pug
        celltype = self._reader.read_array(elem_nodes[0])[0]
        if celltype == 20 : #Mixed
            cell_types, cell_sizes, cells = self._create_ug_mixed(elem_nodes)
        else :
            cell_types, cell_sizes, cells = self._create_ug_notmixed(elem_nodes[0], celltype)
        offset = np.insert(np.cumsum(cell_sizes), 0, 0)
        ug.SetCells(*_numpy_to_cell_array(cell_types, offset, cells))
        return pug

    def _grid_location(self, node):
        r = "Vertex"
        gl = cgns.child_with_label(node, "GridLocation_t")
        if len(gl) > 0:
            r = self._reader.read_array(gl[0])
        return r

    def _add_cell_data(self, zone, node, ug):
        if node.label != "FlowSolution_t":
            return
        gl = self._grid_location(node)
        for c in node.children.values():
            if c.dtype == "C1":
                continue
            self._arrayselection.AddArray(c.name)
            if self._arrayselection.ArrayIsEnabled(c.name):
                base = self._reader.nodes_by_labels(["CGNSBase_t"])[0]
                n = cgns.find_node(base, [_CGN(zone.name), _CGN(node.name), _CGN(c.name)])
                a = self._reader.read_array(n)
                if gl == "Vertex":
                    ug.GetPointData().append(a, c.name)
                else:
                    ug.GetCellData().append(a, c.name)

    def _zone_family(self, zone_node):
        z = cgns.child_with_label(zone_node, "FamilyName_t")
        if not z :
            return None
        return self._reader.read_array(cgns.child_with_label(zone_node, "FamilyName_t")[0])

    def _all_families(self):
        zones = self._reader.nodes_by_labels(["CGNSBase_t", "Zone_t"])
        l = {self._zone_family(z) for z in zones if z is not None}
        if None in l:
            return None
        else:
            return sorted(list(l))

    def _add_iterative_flow_sol(self, zonenode, timeid, ug):
        fsp = cgns.find_node(zonenode, [_CGN("ZoneIterativeData"), _CGN("FlowSolutionPointers")])
        fsps = self._reader.read_array(fsp)
        flowsolutionid = None if fsps is None else fsps[timeid]
        if flowsolutionid is not None:
            flowsol = cgns.find_node(zonenode, [_CGN(flowsolutionid)])
            self._add_cell_data(zonenode, flowsol, ug)

    def _request_iter_data(self, timesteps, outInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet
        data_time = self._get_update_time(outInfoVec.GetInformationObject(0))
        timeid = np.searchsorted(timesteps, data_time)
        bid_node = self._reader.nodes_by_labels(["CGNSBase_t", "BaseIterativeData_t"])
        zonepointers = self._reader.read_array(cgns.child_with_name(bid_node,"ZonePointers"))
        if zonepointers is None:
            # No ZonePointers, we assume one Zone by time step
            zones = self._reader.nodes_by_labels(["CGNSBase_t", "Zone_t"])
            zonelist = [zones[timeid].name]
        else:
            zonelist = zonepointers[timeid]
        base = self._reader.nodes_by_labels(["CGNSBase_t"])[0]
        # switch from Zone names to Zone nodes
        zonelist = [cgns.child_with_name(base, z) for z in zonelist if z is not None]
        # The ZonePointer node of the CGNS file mays contains bullshit so clean
        zonelist = [z for z in zonelist if z is not None]
        mbds = vtkMultiBlockDataSet.GetData(outInfoVec, 0)
        mbds.GetInformation().Set(mbds.DATA_TIME_STEP(), data_time)
        self._np_arrays = []
        fams = self._all_families()
        if fams is None:
            for iz, zonenode in enumerate(zonelist):
                ug = self._create_unstructured_grid(zonenode.name)
                mbds.SetBlock(iz, ug.VTKObject)
                mbds.GetMetaData(iz).Set(mbds.NAME(), zonenode.name)
                self._add_iterative_flow_sol(zonenode, timeid, ug)
        else:
            for ifam, fam in enumerate(fams):
                famblock = vtkMultiBlockDataSet()
                mbds.SetBlock(ifam, famblock)
                mbds.GetMetaData(ifam).Set(mbds.NAME(), fam)
                iz = 0
                for zonenode in zonelist:
                    if self._zone_family(zonenode) != fam:
                        continue
                    ug = self._create_unstructured_grid(zonenode.name)
                    famblock.SetBlock(iz, ug.VTKObject)
                    famblock.GetMetaData(iz).Set(famblock.NAME(), zonenode.name)
                    iz += 1
                    self._add_iterative_flow_sol(zonenode, timeid, ug)
        return 1

    def _request_noiter_data(self, outInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet

        mbds = vtkMultiBlockDataSet.GetData(outInfoVec, 0)
        iz = 0
        for zone in self._reader.nodes_by_labels(["CGNSBase_t", "Zone_t"]):
            ug = self._create_unstructured_grid(zone.name)
            mbds.SetBlock(iz, ug.VTKObject)
            mbds.GetMetaData(iz).Set(mbds.NAME(), zone.name)
            iz += 1
            for flowsol in zone.children.values():
                self._add_cell_data(zone, flowsol, ug)
        return 1

    def RequestData(self, request, inInfoVec, outInfoVec):
        timesteps = self._get_timesteps()
        if timesteps is None:
            self._request_noiter_data(outInfoVec)
        else:
            self._request_iter_data(timesteps, outInfoVec)
        return 1

def test(fname):
    reader = PythonCNGSReader()
    reader.SetFileName(fname)
    from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
    from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataWriter

    writer = vtkXMLMultiBlockDataWriter()
    writer.SetFileName("debug.vtm")
    writer.SetInputConnection(reader.GetOutputPort())
    writer.SetDataModeToAscii()
    writer.Update()
    sf = vtkDataSetSurfaceFilter()
    sf.SetInputConnection(reader.GetOutputPort())
    sf.Update()
    print(sf.GetOutputDataObject(0))


if __name__ == "__main__":
    test("test.cgns")
    from paraview.detail.pythonalgorithm import get_plugin_xmls

    for xml in get_plugin_xmls(globals()):
        print(xml)
