
_base_ = [
    '_base_/default_runtime.py',
    '_base_/datasets/potsdam.py',
    '_base_/models/vimm_small_pot.py',
    '_base_/schedules/vimms_pot_scratch.py',
]
work_dir = './segresults_new/compare/vimmsmall_pot/test/del_h'
