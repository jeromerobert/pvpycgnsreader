import cgns
import numpy as np
from cgns import _CGN
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase


# Map CGNS type to (VTK type, VTK num vertex)
# according to IO/CGNS/vtkCGNSReaderInternal.cxx and Common/DataModel/vtkCellType.h
CELL_TYPE = {
    7: (9, 4),  # QUAD_4 / VTK_QUAD
    5: (5, 3),  # TRI_3 / VTK_TRIANGLE
    17: (12, 8),  # HEXA_8 / VTK_HEXAHEDRON
    12: (14, 5),  # PYRA_5 / VTK_PYRAMID
    14: (13, 6),  # PENTA_6 / VTK_WEDGE
    10: (10, 4),  # TETRA_4 / VTK_TETRA
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


class PythonCNGSReader(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=1, outputType="vtkMultiBlockDataSet"
        )
        self._filename = None
        from vtkmodules.vtkCommonCore import vtkDataArraySelection

        self._arrayselection = vtkDataArraySelection()
        self._modified = False
        self._reader = None
        self._np_arrays = []
        self._current_time = None

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

    def _get_update_time(self, out_info):
        executive = self.GetExecutive()
        timesteps = self._get_timesteps()
        if timesteps is None or len(timesteps) == 0:
            return None
        elif out_info.Has(executive.UPDATE_TIME_STEP()) and len(timesteps) > 0:
            utime = out_info.Get(executive.UPDATE_TIME_STEP())
            dtime = timesteps[0]
            for atime in timesteps:
                if atime > utime:
                    return dtime
                else:
                    dtime = atime
            return dtime
        elif self._current_time is not None:
            return self._current_time
        else:
            assert len(timesteps) > 0
            return timesteps[0]

    def GetTimestepValues(self):
        return self._get_timesteps()

    def RequestInformation(self, request, in_info_vec, out_info_vec):
        timesteps = self._get_timesteps()
        executive = self.GetExecutive()
        out_info = out_info_vec.GetInformationObject(0)
        if timesteps is not None:
            out_info.Set(executive.TIME_STEPS(), timesteps, len(timesteps))
            out_info.Set(executive.TIME_RANGE(), timesteps[[0, -1]], 2)
        return 1

    def _read_grid_coordinates(self, zone):
        # FIXME: allocate the return array and fill it to avoid a memory peak
        r = []
        base = self._reader.nodes_by_labels(["CGNSBase_t"])[0]
        for l in ["X", "Y", "Z"]:
            coord = cgns.find_node(
                base, [_CGN(zone), "GridCoordinates_t", _CGN("Coordinate" + l)]
            )
            a = self._reader.read_array(coord)
            r.append(a.astype(np.single).reshape(-1, 1))
        return np.hstack(r)

    def _create_ug_mixed(self, elem_node):
        c = cgns.child_with_name(elem_node, "ElementConnectivity")
        c_offset = cgns.child_with_name(elem_node, "ElementStartOffset")
        offsets_cgns = self._reader.read_array(c_offset)
        num_cells = len(offsets_cgns) - 1
        cells_cgns = self._reader.read_array(c)
        types_elem = cells_cgns[offsets_cgns[:-1]]
        types_elem_vtk = np.zeros(num_cells, dtype=np.ubyte)
        cells_sizes = np.zeros(num_cells, dtype=int)
        for cgns_type, (vtk_ctype, vtk_csize) in CELL_TYPE.items():
            elem = types_elem == cgns_type
            types_elem_vtk[elem] = vtk_ctype
            cells_sizes[elem] = vtk_csize
        # The VTK cells array is the CGNS connectivity array without the cell types
        mask = np.ones_like(cells_cgns, dtype=bool)
        mask[offsets_cgns[:-1]] = False
        cells_vtk = cells_cgns[mask] - 1
        return types_elem_vtk, cells_sizes, cells_vtk

    def _create_ug_basic(self, element_node, celltype):
        c = cgns.child_with_name(element_node, "ElementConnectivity")
        cells = self._reader.read_array(c) - 1
        cellsize = CELL_TYPE[celltype][1]
        ncells = cells.shape[0] // cellsize
        celltypes = np.full((ncells,), CELL_TYPE[celltype][0], dtype=np.ubyte)
        cellsizes = np.full((ncells,), cellsize, dtype=int)
        return celltypes, cellsizes, cells

    # TODO: vectorize with numpy to improve performance
    def _create_ug_ngon(self, elem_ngon, elem_nface):
        c_ngon = cgns.child_with_name(elem_ngon, "ElementConnectivity")
        c_nfaces = cgns.child_with_name(elem_nface, "ElementConnectivity")
        c_ngon_offset = cgns.child_with_name(elem_ngon, "ElementStartOffset")
        range_ngon = cgns.child_with_name(elem_ngon, "ElementRange")
        offsets_ngon_cgns = self._reader.read_array(c_ngon_offset)
        cells_ngon_cgns = self._reader.read_array(c_ngon) - 1
        cells_nface_cgns = self._reader.read_array(c_nfaces)
        elem_range_ngon = self._reader.read_array(range_ngon)
        cells_vtk = []
        cells_sizes = []
        types_vtk = []
        o_faces = cgns.child_with_name(elem_nface, "ElementStartOffset")
        offset_faces = self._reader.read_array(o_faces)
        num_cells = len(offset_faces) - 1
        num_prev_faces = offset_faces[0]
        for i in range(1, num_cells + 1):  # loop on cells
            cells_size = 0
            num_faces = offset_faces[i] - num_prev_faces
            cells_vtk.append(num_faces)
            cells_size += 1
            list_ind_faces = [
                cells_nface_cgns[k]
                for k in range(num_prev_faces, num_prev_faces + num_faces)
            ]
            for face in list_ind_faces:  # loop on faces
                if face >= 0:
                    face = face - elem_range_ngon[0]
                else:
                    face = -(abs(face) - elem_range_ngon[0])
                num_nodes = (
                    offsets_ngon_cgns[abs(face) + 1] - offsets_ngon_cgns[abs(face)]
                )
                cells_size += num_nodes + 1
                cells_vtk.append(num_nodes)
                ind_nodes = []
                beggining_of_face = offsets_ngon_cgns[abs(face)]
                for j in range(num_nodes):  # loop on nodes
                    ind_nodes.append(cells_ngon_cgns[beggining_of_face + j])
                if face < 0:
                    ind_nodes.reverse()
                for k in range(num_nodes):
                    cells_vtk.append(ind_nodes[k])
            cells_sizes.append(cells_size)
            types_vtk.append(42)
            num_prev_faces = offset_faces[i]

        return types_vtk, cells_sizes, cells_vtk

    def _find_ngon_from_nface(self, elem_nodes):
        for elem_node in elem_nodes:
            celltype = self._reader.read_array(elem_node)[0]
            if celltype == 22:
                return elem_node

    def _create_unstructured_grid(self, zone):
        from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
        from vtkmodules.numpy_interface import dataset_adapter as dsa

        ug = vtkUnstructuredGrid()
        pug = dsa.WrapDataObject(ug)
        pug.SetPoints(self._read_grid_coordinates(zone.name))
        elem_nodes = cgns.child_with_label(zone, "Elements_t")
        data = []  # VTK (cell_types, cell_sizes, cells) for each Element_t

        for elem_node in elem_nodes:
            celltype = self._reader.read_array(elem_node)[0]
            if celltype == 23:
                elem_nface = elem_node
                elem_ngon = self._find_ngon_from_nface(elem_nodes)
                data.append(self._create_ug_ngon(elem_ngon, elem_nface))
                del elem_nodes[elem_nodes.index(elem_nface)]
                del elem_nodes[elem_nodes.index(elem_ngon)]

        # FIXME: must iterate on Element_t in the "ElementRange" order
        for elem_node in elem_nodes:
            celltype = self._reader.read_array(elem_node)[0]
            if celltype == 20:  # Mixed
                data.append(self._create_ug_mixed(elem_node))
            else:
                data.append(self._create_ug_basic(elem_node, celltype))
        cell_types = np.concatenate([x[0] for x in data])
        cell_sizes = np.concatenate([x[1] for x in data])
        cells = np.concatenate([x[2] for x in data])
        offset = np.insert(np.cumsum(cell_sizes), 0, 0)
        ug.SetCells(*_numpy_to_cell_array(cell_types, offset, cells))
        return pug

    def _grid_location(self, node):
        r = "Vertex"
        gl = cgns.child_with_label(node, "GridLocation_t")
        if len(gl) > 0:
            r = self._reader.read_array(gl[0])
        return r

    def _find_vector_data(self, data_names):
        vec_labels = []
        vec_names = set()
        for name in data_names:
            minus = name.endswith("x")
            maj = name.endswith("X")
            if not (minus or maj):
                continue
            sep = ""
            if len(name) > 1 and name[-2] in ["_", " "]:
                sep = name[-2]
            p = name[: -(len(sep) + 1)]
            y = p + sep + ("y" if minus else "Y")
            z = p + sep + ("z" if minus else "Z")
            if y in data_names and z in data_names:
                vec_labels.append((p, [name, y, z]))
                vec_names.update([name, y, z])
        return [n for n in data_names if n not in vec_names], vec_labels

    def _add_cell_data(self, zone, node, ug):
        if node.label != "FlowSolution_t":
            return
        base = self._reader.nodes_by_labels(["CGNSBase_t"])[0]
        vertex_data = self._grid_location(node) == "Vertex"
        data = ug.GetPointData() if vertex_data else ug.GetCellData()
        data_names = []
        for c in node.children.values():
            if c.dtype == "C1":
                continue
            self._arrayselection.AddArray(c.name)
            if self._arrayselection.ArrayIsEnabled(c.name):
                data_names.append(c.name)
        data_names, vec_labels = self._find_vector_data(data_names)
        for name in data_names:
            n = cgns.find_node(base, [_CGN(zone.name), _CGN(node.name), _CGN(name)])
            data.append(self._reader.read_array(n), name)
        for name, comps in vec_labels:
            a = []
            for c in comps:
                n = cgns.find_node(base, [_CGN(zone.name), _CGN(node.name), _CGN(c)])
                a.append(self._reader.read_array(n))
            data.append(np.vstack(a).T, name)

    def _zone_family(self, zone_node):
        z = cgns.child_with_label(zone_node, "FamilyName_t")
        if not z:
            return None
        return self._reader.read_array(
            cgns.child_with_label(zone_node, "FamilyName_t")[0]
        )

    def _all_families(self):
        zones = self._reader.nodes_by_labels(["CGNSBase_t", "Zone_t"])
        l = {self._zone_family(z) for z in zones if z is not None}
        if None in l:
            return None
        else:
            return sorted(list(l))

    def _add_iterative_flow_sol(self, zonenode, timeid, ug):
        fsp = cgns.find_node(
            zonenode, [_CGN("ZoneIterativeData"), _CGN("FlowSolutionPointers")]
        )
        fsps = self._reader.read_array(fsp)
        flowsolutionid = None if fsps is None else fsps[timeid]
        if flowsolutionid is not None:
            flowsol = cgns.find_node(zonenode, [_CGN(flowsolutionid)])
            self._add_cell_data(zonenode, flowsol, ug)

    def _request_iter_data(self, timesteps, out_info_vec):
        from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet

        data_time = self._get_update_time(out_info_vec.GetInformationObject(0))
        timeid = np.searchsorted(timesteps, data_time)
        bid_node = self._reader.nodes_by_labels(["CGNSBase_t", "BaseIterativeData_t"])
        zonepointers = self._reader.read_array(
            cgns.child_with_name(bid_node, "ZonePointers")
        )
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
        mbds = vtkMultiBlockDataSet.GetData(out_info_vec, 0)
        mbds.GetInformation().Set(mbds.DATA_TIME_STEP(), data_time)
        self._np_arrays = []
        fams = self._all_families()
        if fams is None:
            for iz, zonenode in enumerate(zonelist):
                ug = self._create_unstructured_grid(zonenode)
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
                    ug = self._create_unstructured_grid(zonenode)
                    famblock.SetBlock(iz, ug.VTKObject)
                    famblock.GetMetaData(iz).Set(famblock.NAME(), zonenode.name)
                    iz += 1
                    self._add_iterative_flow_sol(zonenode, timeid, ug)
        return 1

    def _request_noiter_data(self, out_info_vec):
        from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet

        mbds = vtkMultiBlockDataSet.GetData(out_info_vec, 0)
        iz = 0
        for zone in self._reader.nodes_by_labels(["CGNSBase_t", "Zone_t"]):
            ug = self._create_unstructured_grid(zone)
            mbds.SetBlock(iz, ug.VTKObject)
            mbds.GetMetaData(iz).Set(mbds.NAME(), zone.name)
            iz += 1
            for flowsol in zone.children.values():
                self._add_cell_data(zone, flowsol, ug)
        return 1

    def RequestData(self, request, in_info_vec, out_info_vec):
        timesteps = self._get_timesteps()
        if timesteps is None:
            self._request_noiter_data(out_info_vec)
        else:
            self._request_iter_data(timesteps, out_info_vec)
        return 1

    def SetTimeStep(self, t):
        """Set the current time step"""
        executive = self.GetExecutive()
        reader_info = executive.GetOutputInformation(0)
        reader_info.Set(executive.UPDATE_TIME_STEP(), t)
        self._current_time = t


def test(fname):
    from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataWriter

    reader = PythonCNGSReader()
    reader.SetFileName(fname)
    writer = vtkXMLMultiBlockDataWriter()
    writer.SetInputConnection(reader.GetOutputPort())
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    writer.SetCompressorTypeToNone()
    timesteps = reader.GetTimestepValues()
    for i, t in enumerate(timesteps):
        reader.SetTimeStep(t)
        writer.SetFileName(f"testcgns_{i}.vtm")
        writer.Write()


if __name__ == "__main__":
    test("test.cgns")
