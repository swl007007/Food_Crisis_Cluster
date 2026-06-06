import numpy as np

from src.helper.helper import get_X_branch_id_by_group, init_s_branch


def test_init_s_branch_preserves_active_non_contiguous_polygon_group_ids():
    active_group_ids = np.array([732, 1000, 1015], dtype=np.int32)

    s_branch, _ = init_s_branch(
        n_groups=len(active_group_ids),
        max_depth=2,
        group_ids=active_group_ids,
    )

    assert s_branch[""].to_numpy()[:3].tolist() == [732, 1000, 1015]

    mapped = get_X_branch_id_by_group(
        np.array([1015, 732, 1000], dtype=np.int32),
        s_branch,
        max_depth=2,
    )

    assert mapped.tolist() == ["", "", ""]


def test_init_s_branch_dense_grid_ids_do_not_overflow_int16_boundary():
    s_branch, _ = init_s_branch(n_groups=32769, max_depth=1)

    values = s_branch[""].to_numpy()

    assert int(values[32768]) == 32768
