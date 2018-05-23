def test_recon_carved():
    elec_ind = 1
    other_inds = [i for i in range(sub_locs.shape[0]) if i != elec_ind]
    bo_t = bo.get_slice(loc_inds=other_inds, inplace=False)
    bo_r = test_model.predict(bo_t, nearest_neighbor=False)
    bo_l = bo.get_slice(loc_inds=elec_ind , inplace=False)

    mo_inds = np.where(_count_overlapping(test_model.get_locs(), bo_l.get_locs()))[0]
    bo_inds = np.where(_count_overlapping(bo_r.get_locs(), bo_l.get_locs())==1)[0]
    bo_p = bo_r.get_slice(loc_inds=bo_inds, inplace=False)

    mo_carve_inds = _count_overlapping(test_model.get_locs(), bo.get_locs())
    carved_mat = test_model.get_slice(inds=mo_carve_inds)
    bo_c = carved_mat.predict(bo_t, nearest_neighbor=False)
    bo_carve_inds = _count_overlapping(bo_c.get_locs(), bo_l.get_locs())
    bo_p_2 = bo_c.get_slice(loc_inds=bo_carve_inds, inplace=False)
    assert np.allclose(bo_p.get_data(), bo_p_2.get_data())
