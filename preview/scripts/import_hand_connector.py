#!/usr/bin/env python3
"""Tessellate the supplied STEP assembly, retaining every instance transform.

Run with the optional CAD environment (cadquery-ocp 7.8.1.1.post1).
Output vertices use metres. The reference hand is evidence for registration,
not a replacement for the articulated URDF hand.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDF import TDF_LabelSequence, TDF_Label
    from OCP.TDataStd import TDataStd_Name
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.StlAPI import StlAPI_Writer
    from OCP.gp import gp_Trsf, gp_Pnt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference-product', default='A3手')
    parser.add_argument('--skip-reference-mesh', action='store_true')
    parser.add_argument('--extract-product', action='append', default=[],
                        help='Also export named nested components in assembly coordinates')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(args.step)) != IFSelect_RetDone:
        raise ValueError('STEP import failed')
    doc = TDocStd_Document(TCollection_ExtendedString('connector'))
    if not reader.Transfer(doc):
        raise ValueError('STEP assembly transfer failed')
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = TDF_LabelSequence()
    tool.GetFreeShapes(roots)
    if roots.Length() != 1:
        raise ValueError('Expected one assembly')
    children = TDF_LabelSequence()
    tool.GetComponents_s(roots.Value(1), children)
    args.output.mkdir(parents=True)
    report = {'source_filename': args.step.name, 'source_sha256': hashlib.sha256(args.step.read_bytes()).hexdigest(),
              'source_units': 'mm', 'mesh_units': 'm', 'linear_tessellation_tolerance_mm': .1,
              'flange_frame': 'STEP assembly origin; circular adapter top face z=0', 'instances': []}
    scale = gp_Trsf()
    scale.SetScale(gp_Pnt(0, 0, 0), .001)
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    def product_name(label):
        attr = TDataStd_Name()
        if not label.FindAttribute(TDataStd_Name.GetID_s(), attr):
            return ''
        return attr.Get().ToExtString()

    def export_shape(shape, filename):
        shape = BRepBuilderAPI_Transform(shape, scale, True).Shape()
        BRepMesh_IncrementalMesh(shape, .0001, False, .25, True).Perform()
        if not writer.Write(shape, str(args.output / filename)):
            raise ValueError('STL export failed')
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        return {'mesh': filename, 'bounds_m': list(box.Get()),
                'mesh_sha256': hashlib.sha256((args.output / filename).read_bytes()).hexdigest()}

    for index in range(1, children.Length() + 1):
        instance = children.Value(index)
        reference = TDF_Label()
        if not tool.GetReferredShape_s(instance, reference):
            raise ValueError('Expected assembly component reference')
        name = product_name(reference)
        shape = tool.GetShape_s(instance)
        exported = {} if name == args.reference_product and args.skip_reference_mesh else export_shape(shape, f'part_{index:02d}.stl')
        transform = tool.GetLocation_s(instance).Transformation()
        report['instances'].append({'product': name, **exported, 'is_reference_hand': name == args.reference_product,
            'assembly_from_part_mm': [[transform.Value(r, c) for c in range(1, 5)] for r in range(1, 4)],
            })
    extracted = []
    def visit(label, parent_transform):
        seq = TDF_LabelSequence()
        tool.GetComponents_s(label, seq)
        for i in range(1, seq.Length() + 1):
            instance = seq.Value(i)
            reference = TDF_Label()
            if not tool.GetReferredShape_s(instance, reference):
                raise ValueError('Expected nested component reference')
            transform = parent_transform.Multiplied(tool.GetLocation_s(instance).Transformation())
            name = product_name(reference)
            if name in args.extract_product:
                shape = BRepBuilderAPI_Transform(tool.GetShape_s(reference), transform, True).Shape()
                exported = export_shape(shape, f'extracted_{len(extracted)+1:02d}.stl')
                extracted.append({'product': name, **exported,
                    'assembly_from_part_mm': [[transform.Value(r, c) for c in range(1, 5)] for r in range(1, 4)]})
            if tool.IsAssembly_s(reference):
                visit(reference, transform)
    if args.extract_product:
        visit(roots.Value(1), gp_Trsf())
        if set(args.extract_product) != {part['product'] for part in extracted}:
            raise ValueError('Requested nested product was not found')
        report['extracted_components'] = extracted
    (args.output / 'assembly.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'instances': children.Length(), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
