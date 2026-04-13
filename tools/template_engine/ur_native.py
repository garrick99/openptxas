"""UR-native address generation path design and trigger rules.

TEMPLATE-ENGINE-9A: bounded UR-native execution strategy for uniform
param/bounds-check kernels.

Target: 33 kernels at instruction parity with UR-compare gap.
Strategy: replace u32 param LDC-to-GPR with LDCU.32-to-UR (1:1 swap),
enabling ISETP.R-UR instead of ISETP.R-R.

Trigger rules (ALL must hold):
1. Kernel is SM_120
2. u32 param is consumed ONLY by setp (no other GPR consumers)
3. Kernel does NOT have VOTE instructions (ISETP.UR + VOTE = ERR715)
4. Kernel does NOT have atom.xor template active
5. No BAR.SYNC in kernel (body LDCU hazard)
"""

UR_NATIVE_TRIGGER_RULES = [
    {'rule': 'SM_120', 'reason': 'UR-native path only implemented for Blackwell'},
    {'rule': 'setp_only_param', 'reason': 'param must not have non-setp GPR consumers'},
    {'rule': 'no_vote', 'reason': 'ISETP.UR + VOTE causes ERR715 on SM_120'},
    {'rule': 'no_atom_xor_template', 'reason': 'atom.xor uses dedicated template path'},
    {'rule': 'no_bar_sync', 'reason': 'BAR kernels have body LDCU hazards'},
]

TARGET_FAMILY = {
    'name': 'uniform_bounds_check',
    'description': 'Simple param/bounds-check kernels with out[tid]=f(tid) pattern',
    'expected_impact': '33 kernels at instruction parity gain ISETP.R-UR',
    'risk': 'LOW — 1:1 instruction substitution, no extra instructions',
}
