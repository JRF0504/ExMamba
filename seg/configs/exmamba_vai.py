
_base_ = [
    '_base_/default_runtime.py',
    '_base_/datasets/vaihingen.py',
    '_base_/models/vimm_small_vai.py',
    '_base_/schedules/vimms_vai_scratch.py',
]
work_dir = './segresults_new/compare/vimmsmall_vai/test/del_linear'

