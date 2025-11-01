using Test

_validate_args((100, 200); nwindows=(3, 4), window_size=(10, 10), window_step=(), relative_overlap=())

_validate_args((100, 200); nwindows=(3,), window_size=(10, 10), window_step=(), relative_overlap=())
# ERROR: ArgumentError: Length of nwindows (1) must match npoints (2)