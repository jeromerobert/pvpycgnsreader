from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain, smhint
import sys
import os

sys.path.append(os.path.dirname(__file__))
# Flush the cgns module when paraview reload the module
sys.modules.pop("cngs", None)
sys.modules.pop("pycgnsreader", None)
from pycgnsreader import PythonCNGSReader


def modified_callback(anobject):
    import weakref

    weakref_obj = weakref.ref(anobject)
    anobject = None

    def _markmodified(*args, **kwars):
        o = weakref_obj()
        if o is not None:
            o.Modified()

    return _markmodified


@smproxy.reader(
    name="PVPythonCNGSReader",
    label="Python-based CGNS Reader",
    extensions="cgns",
    file_description="CGNS files",
)
class PVPythonCNGSReader(PythonCNGSReader):
    def __init__(self):
        super().__init__()
        self._arrayselection.AddObserver("ModifiedEvent", modified_callback(self))

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions="cgns", file_description="CGNS files")
    def SetFileName(self, name):
        super().SetFileName(name)

    @smproperty.doublevector(
        name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty"
    )
    def GetTimestepValues(self):
        super().GetTimestepValues()

    # Array selection API is typical with readers in VTK
    # This is intended to allow ability for users to choose which arrays to
    # load. To expose that in ParaView, simply use the
    # smproperty.dataarrayselection().
    # This method **must** return a `vtkDataArraySelection` instance.
    @smproperty.dataarrayselection(name="Arrays")
    def GetDataArraySelection(self):
        return self._arrayselection


def test(fname):
    reader = PVPythonCNGSReader()
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
