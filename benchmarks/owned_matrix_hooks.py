"""Restore only our temporary instance hooks before dropping an optimizer."""
_MISSING = object()
HOOKS = ('_research_refresh', '_gauge_step_one')
GROUP_KEYS = ('research_movement_policy', 'basis_ns_iterations',
              'basis_rotation_cap', 'basis_rotation_cap_mode')


class OwnedMatrixHooks:
    def __init__(self, matrix):
        self.matrix = matrix
        self.previous = {key: matrix.__dict__.get(key, _MISSING) for key in HOOKS}
        self.groups = [{key: group.get(key, _MISSING) for key in GROUP_KEYS}
                       for group in matrix.param_groups]

    def close(self):
        """Restore instance topology, avoiding a stored bound class method cycle."""
        if self.matrix is None:
            return
        matrix = self.matrix
        for key, value in self.previous.items():
            if value is _MISSING:
                matrix.__dict__.pop(key, None)
            else:
                matrix.__dict__[key] = value
        for group, original in zip(matrix.param_groups, self.groups, strict=True):
            for key, value in original.items():
                if value is _MISSING:
                    group.pop(key, None)
                else:
                    group[key] = value
        self.matrix = None
        self.previous.clear()
        self.groups.clear()
