# -*- coding:utf-8 -*-
# author: Xinge
# @file: model_builder.py 
import sys
import os
GRANDFA = os.path.dirname(os.path.realpath(__file__))
sys.path.append(GRANDFA)

from segmentator_3d_asymm_spconv import Asymm_3d_spconv
from cylinder_fea_generator import cylinder_fea
from cylinder_spconv_3d import get_model_class

def cylinder_build():
    output_shape = [480, 360, 32]
    num_class = 20
    num_input_features = 16
    use_norm = True
    init_size = 32
    fea_dim = 9
    out_fea_dim = 256

    cylinder_3d_spconv_seg = Asymm_3d_spconv(
        output_shape=output_shape,
        use_norm=use_norm,
        num_input_features=num_input_features,
        init_size=init_size,
        nclasses=num_class)

    cy_fea_net = cylinder_fea(grid_size=output_shape,
                              fea_dim=fea_dim,
                              out_pt_fea_dim=out_fea_dim,
                              fea_compre=num_input_features)

    model = get_model_class("cylinder_asym")(
        cylin_model=cy_fea_net,
        segmentator_spconv=cylinder_3d_spconv_seg,
        sparse_shape=output_shape
    )

    return model
