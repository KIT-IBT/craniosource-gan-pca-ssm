"""
Create images from SSM.
"""

import argparse
import os.path
import pathlib
import pickle

import vtk
import numpy
import numpy.random
import skimage.io

import distmaps
import painting

def parseargs():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i',
                        '--input_dir',
                        type=str,
                        help="Input directory",
                        default=os.path.expanduser("../ssm"))
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help="Output directory",
                        default=os.path.expanduser("../demo_out/ssm"))
    parser.add_argument('-n',
                        '--number',
                        type=int,
                        help="Number images",
                        default=10)
    arguments = parser.parse_args()
    return arguments

def columnvec2xyz(observations):
    """
    Reshape a columnvector [x1, y1, z1, x2, ... ] into a 2D vector of 
    [[x1, y1, z1], [x2, ...], ...]
    """
    axis = 0
    cur_shape = numpy.array(observations).shape
    assert cur_shape[-1] % 3 == 0, f"last axis {axis} not divisible by 3"
    if len(cur_shape) == 1:
        return numpy.reshape(observations,(cur_shape[0]//3,3))
    my_list = list(cur_shape)
    my_list[-1] //= 3
    my_list += [3]
    return numpy.reshape(observations,tuple(my_list))

def polydata_from_points_cells(points,cells):
    """
    Converts points and cells to vtk.vtkPolydata.
    ----------
    Parameters
    points : [Px3 numpy array] 
    cells : [Cx3 numpy array]
    ----------
    Returns
    polydata : vtk.vtkPolydata
    """
    # This function follows mostly this example:
    # https://kitware.github.io/vtk-examples/site/Python/GeometricObjects/Triangle/
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point[0],point[1],point[2])
    vtk_cells = vtk.vtkCellArray()
    for cell in cells:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, cell[0])
        triangle.GetPointIds().SetId(1, cell[1])
        triangle.GetPointIds().SetId(2, cell[2])
        vtk_cells.InsertNextCell(triangle)
    # Set polydata
    my_polydata = vtk.vtkPolyData()
    my_polydata.SetPoints(vtk_points)
    my_polydata.SetPolys(vtk_cells)
    return my_polydata

if __name__ == "__main__":
    args = parseargs()
    with open(f"{args.input_dir}/template_cells.pickle", "rb") as handle:
        cls_t = pickle.load(handle)
    classes = ("control", "coronal", "metopic", "sagittal")
    painter = painting.ImageFromDistancesCreator()
    for cls in classes:
        with open(f"{args.input_dir}/{cls}.pickle", "rb") as handle:
            sm_dict = pickle.load(handle)
        cur_dir = args.output_dir + f"/{cls}"
        if not os.path.exists(cur_dir):
            pathlib.Path(cur_dir).mkdir(parents=True)
        for i in range(args.number):
            n_components = len(sm_dict['eigenvalues'])
            alpha = numpy.random.normal(size=n_components)
            stat = (sm_dict['eigenvalues'] ** 0.5 * alpha) @ sm_dict['eigenvectors']
            shp = columnvec2xyz(sm_dict['mu'] + stat)
            polydata = polydata_from_points_cells(shp,cls_t)
            lms_inds = distmaps.define_shape_model_landmarks()
            lms_10 = shp[lms_inds]
            cdl = lms_10[distmaps.define_center_landmark_ids(),:]
            intersector = distmaps.MeshIntersector(polydata,cdl)
            creator = distmaps.RayCreator(method="halfsphere",intersector=intersector)
            distances = creator()
            img = painter([distances])[0]
            img_uint8 = numpy.uint8(img * 255)
            skimage.io.imsave(cur_dir + f"/drawn_{i}.png",img_uint8)
            print(f"Done {cls} {i}")
