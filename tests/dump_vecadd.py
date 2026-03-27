"""
Dump the full SASS instruction sequence for vector_add, including
ctrl words and register assignments.
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass.pipeline import compile_ptx_source, compile_function, _sink_param_loads
from sass.regalloc import allocate
from sass.isel import ISelContext, select_function, SassInstr
from sass.scoreboard import assign_ctrl
from sass.schedule import schedule
from sass.encoding.sm_120_opcodes import encode_ldcu_64, encode_s2r, SR_TID_X, encode_bra
from ptx.parser import parse
from ptx.ir import RegOp

_PTX = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry vector_add(
    .param .u64 out, .param .u64 a, .param .u64 b, .param .u32 n)
{
    .reg .b32 %r<8>; .reg .b64 %rd<8>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;
    cvt.u64.u32 %rd0, %r3;
    shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [a]; add.u64 %rd2, %rd1, %rd0;
    ld.param.u64 %rd3, [b]; add.u64 %rd4, %rd3, %rd0;
    ld.global.u32 %r5, [%rd2];
    ld.global.u32 %r6, [%rd4];
    add.s32 %r7, %r5, %r6;
    ld.param.u64 %rd5, [out]; add.u64 %rd6, %rd5, %rd0;
    st.global.u32 [%rd6], %r7;
DONE:
    ret;
}
"""


def decode_ctrl(raw16: bytes) -> dict:
    b13 = raw16[13]; b14 = raw16[14]; b15 = raw16[15]
    raw24 = ((b15 & ~0x04) << 16) | (b14 << 8) | b13
    ctrl = raw24 >> 1
    stall = (ctrl >> 17) & 0x3f
    rbar  = (ctrl >> 10) & 0x1f
    wdep  = (ctrl >>  4) & 0x3f
    misc  = ctrl & 0xf
    return {'stall': stall, 'rbar': rbar, 'wdep': wdep, 'misc': misc}


# Parse and run pipeline stages manually to get annotated output
mod = parse(_PTX)
fn = mod.functions[0]

# Stage 0: sink param loads
_sink_param_loads(fn)

# Stage 1: regalloc
alloc = allocate(fn)
ra = alloc.ra

print(f"=== Register Allocation ===")
print(f"  int_regs: {dict(ra.int_regs)}")
print(f"  pred_regs: {ra.pred_regs}")
print(f"  num_gprs: {alloc.num_gprs}")
print(f"  param_offsets: {dict((k, hex(v)) for k,v in alloc.param_offsets.items())}")

# Stage 2: ISEL
ctx = ISelContext(
    ra=alloc.ra,
    param_offsets=alloc.param_offsets,
    ur_desc=4,
)
body_instrs = select_function(fn, ctx)

# Insert UR4 descriptor load (as pipeline does)
has_s2r = any(
    struct.unpack_from('<Q', si.raw, 0)[0] & 0xFFF in (0x919, 0x9c3)
    for si in body_instrs
)
if not has_s2r:
    body_instrs.insert(0, SassInstr(encode_s2r(0, SR_TID_X),
                                     'S2R R0, SR_TID.X  // required'))

insert_idx = 0
for idx, si in enumerate(body_instrs):
    opcode = struct.unpack_from('<Q', si.raw, 0)[0] & 0xFFF
    if opcode in (0x919, 0x9c3):
        insert_idx = idx + 1
ur4_desc = SassInstr(encode_ldcu_64(4, 0, 0x358, ctrl=0x717),
                      'LDCU.64 UR4, c[0][0x358]  // mem desc')
body_instrs.insert(insert_idx, ur4_desc)

preamble = [SassInstr(bytes.fromhex('827b01ff00df00000008000000e20f00'),
                       'LDC R1, c[0][0x37c]  // frame ptr')]

print(f"\n=== ISEL output ({len(body_instrs)} body instrs before schedule) ===")
for i, si in enumerate(body_instrs):
    raw = si.raw
    opcode = struct.unpack_from('<Q', raw)[0] & 0xFFF
    print(f"  [{i:2d}] 0x{opcode:03x}  {si.comment}")

# Stage 3: schedule
raw_instrs = preamble + body_instrs
reordered = schedule(raw_instrs)
n_preamble = len(preamble)

# Stage 4: scoreboard
body_scheduled = assign_ctrl(reordered[n_preamble:])
sass_instrs = reordered[:n_preamble] + body_scheduled

print(f"\n=== Final SASS ({len(sass_instrs)} instrs after schedule+scoreboard) ===")
print(f"{'idx':>3}  {'opc':>7}  stall  rbar    wdep  misc  comment")
print('-' * 100)
for i, si in enumerate(sass_instrs):
    raw = si.raw
    opcode = struct.unpack_from('<Q', raw)[0] & 0xFFF
    if i < n_preamble:
        print(f"[{i:2d}] 0x{opcode:03x}  (preamble hardcoded ctrl)  {si.comment}")
    else:
        ctrl = decode_ctrl(raw)
        stall = ctrl['stall']; rbar = ctrl['rbar']; wdep = ctrl['wdep']; misc = ctrl['misc']
        print(f"[{i:2d}] 0x{opcode:03x}  stall={stall:2d}  rbar=0x{rbar:02x}  "
              f"wdep=0x{wdep:02x}  misc=0x{misc:x}  {si.comment}")

# Also show full 32 bytes for critical instructions
print(f"\n=== Full raw bytes for instructions 8-22 ===")
for i in range(min(8, n_preamble), min(len(sass_instrs), 24)):
    raw = sass_instrs[i].raw
    print(f"[{i:2d}] {raw.hex()}  {sass_instrs[i].comment}")
