import React from "react";
import { motion } from "framer-motion";

// ---------------------------------------------------------------------------
// OpenPTXas technical showcase — single-page React component
//
// Dependencies: React, Tailwind CSS, framer-motion
// Drop into a Tailwind-configured React app (Next.js, Vite, CRA) and render
// <OpenPTXasShowcase /> from any route/page.
// ---------------------------------------------------------------------------

// ---------- motion presets ----------
const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } },
};

const stagger = {
  visible: { transition: { staggerChildren: 0.08 } },
};

// ---------- primitives ----------
const Section = ({ id, children, className = "" }) => (
  <motion.section
    id={id}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true, margin: "-80px" }}
    variants={stagger}
    className={`max-w-6xl mx-auto px-6 py-24 ${className}`}
  >
    {children}
  </motion.section>
);

const Card = ({ children, className = "" }) => (
  <motion.div
    variants={fadeUp}
    whileHover={{ y: -4, transition: { duration: 0.2 } }}
    className={`bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-lg shadow-slate-950/50 backdrop-blur-sm ${className}`}
  >
    {children}
  </motion.div>
);

const SectionHeader = ({ eyebrow, title, description, align = "center" }) => (
  <motion.div
    variants={fadeUp}
    className={`mb-14 max-w-3xl ${align === "center" ? "mx-auto text-center" : ""}`}
  >
    {eyebrow && (
      <div className="text-cyan-400 text-xs font-semibold tracking-[0.2em] uppercase mb-3">
        {eyebrow}
      </div>
    )}
    <h2 className="text-4xl md:text-5xl font-bold text-white mb-4 tracking-tight">
      {title}
    </h2>
    {description && (
      <p className="text-lg text-slate-400 leading-relaxed">{description}</p>
    )}
  </motion.div>
);

// ---------- simple inline icons ----------
const IconCheck = ({ className = "w-5 h-5" }) => (
  <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 10 l4 4 l8 -8" />
  </svg>
);

const IconArrow = ({ className = "w-5 h-5" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 4 v16" />
    <path d="M6 14 l6 6 l6 -6" />
  </svg>
);

const IconLightning = ({ className = "w-5 h-5" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M13 2 L4 14 h7 L10 22 L20 10 h-7 z" />
  </svg>
);

const IconShield = ({ className = "w-5 h-5" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round">
    <path d="M12 3 L4 6 v6 c0 5 3.5 8.5 8 9 c4.5 -.5 8 -4 8 -9 V6 z" />
  </svg>
);

const IconCode = ({ className = "w-5 h-5" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M16 18 l6 -6 l-6 -6" />
    <path d="M8 6 l-6 6 l6 6" />
  </svg>
);

// ---------- data ----------
const STACK = [
  {
    name: "Forge",
    role: "Formally-verified systems language",
    detail: "High-level language targeting C99 / CUDA C. Provides the source correctness guarantees that propagate downstream.",
  },
  {
    name: "OpenCUDA",
    role: "CUDA C → PTX",
    detail: "CUDA C front-end producing clean PTX. 31K+ tests, texture & TMA intrinsics, 11 GPU-verified E2E flows.",
  },
  {
    name: "PTX",
    role: "Parallel Thread eXecution IR",
    detail: "NVIDIA's virtual ISA. The contract between front-end and back-end.",
  },
  {
    name: "OpenPTXas",
    role: "PTX → SASS assembler (SM_120)",
    detail: "Native SM_120 assembler. No ptxas fallback path. Instruction selection, register allocation, scheduling, scoreboard, encoding — fully owned.",
    highlight: true,
  },
  {
    name: "SASS / CUBIN",
    role: "Native GPU binary",
    detail: "ELF-wrapped SASS ready for cuModuleLoadData.",
  },
  {
    name: "RTX 5090 (SM_120)",
    role: "Execution target",
    detail: "Verified on live hardware. 142/142 corpus kernels execute with bit-identical correctness.",
  },
];

const CORPUS_CATEGORIES = [
  { name: "ALU chains", note: "IADD3, IMAD, LOP3, SHF, predicated ops" },
  { name: "Predicate logic", note: "ISETP families, @P / @!P guards, fused setp" },
  { name: "Memory operations", note: "LDG / STG, LDC, LDCU, descriptor-based addressing" },
  { name: "Loops", note: "Back-edge BRA with predicate handoff, nested loops" },
  { name: "Warp intrinsics", note: "SHFL, VOTE, REDUX, ballot / sync patterns" },
  { name: "Shared memory + barriers", note: "STS / LDS, BAR.SYNC, BSYNC.RECONVERGENT" },
  { name: "Atomics", note: "ATOMG.ADD / AND / OR / MIN / MAX / EXCH / CAS" },
  { name: "Tensor ops (zero-acc)", note: "HMMA / DMMA / IMMA / QMMA accumulator patterns" },
];

const TIMELINE = [
  {
    tag: "R31",
    title: "UR-route u64 params across predicated EXIT",
    detail: "u64 pointer params redefined in-place survive the early-exit guard cleanly.",
  },
  {
    tag: "R32′",
    title: "WB-8 LDCU.128 pack guard",
    detail: "Post-EXIT R-UR consumers no longer spill into the pack region.",
  },
  {
    tag: "R38 / R39",
    title: "Post-EXIT S2R → ALU gap",
    detail: "Scoreboard doesn't honor S2R-writes-across-EXIT reliably — fixed with a one-instruction gap, extended to UIADD and LOP3.LUT consumers.",
  },
  {
    tag: "R48",
    title: "ISETP → @P adjacency hazard",
    detail: "Direct ISETP-to-predicated-consumer adjacency on SM_120 reads a stale predicate. One NOP-gap pass closed predicated EXIT, predicated ALU, and loop back-edge BRA in a single landing.",
    big: true,
  },
  {
    tag: "R49 / R50",
    title: "Rename-pass tightening",
    detail: "FG29-C: exclude imm-at-b4 opcodes from the rename, preserve 64-bit pair hi-halves in the outside-user scan.",
  },
  {
    tag: "R52",
    title: "ld.param.u64 PTX position preserved",
    detail: "Multi-BB labeled-consumer kernels no longer hoist u64 loads above the setp param, preventing UR alias.",
  },
  {
    tag: "R55 / R56",
    title: "FG26 UR4 admission gated on TE10",
    detail: "Warp-intrinsic and bar.sync kernels now correctly skip UR4 reservation when the setp-UR-native path is blocked.",
  },
  {
    tag: "R57",
    title: "Preamble-hoist GPR alias",
    detail: "Preamble-hoisted LDC params that collide with an SR-derived register are reassigned pre-emit.",
  },
  {
    tag: "R58",
    title: "ATOMG_AND encoding",
    detail: "Corrected mode-table bytes to live ptxas ground truth. Final failure closed — corpus fully green.",
    closing: true,
  },
];

const PRINCIPLES = [
  {
    title: "Proof over assumption",
    detail: "Every fix begins with a minimal probe that reproduces the failure byte-for-byte. No guesses, no drive-by patches.",
  },
  {
    title: "Narrow fixes only",
    detail: "One landing ≈ one proven root cause. Broad refactors are rejected at the gate even when tempting.",
  },
  {
    title: "GPU-verified correctness",
    detail: "Every green is validated on physical silicon (RTX 5090, SM_120) — not just compiler output parity.",
  },
  {
    title: "No regressions",
    detail: "Every push re-runs the full corpus before merge. A fix that costs a prior green is reverted immediately.",
  },
];

const NEXT_PHASE = [
  {
    title: "CI regression gate",
    detail: "Hook scripts/corpus_sweep.py into OpenCUDA's CI pipeline. Any backend regression fails the PR before merge.",
  },
  {
    title: "Performance parity expansion",
    detail: "Apply the SAXPY SASS-diff + timing harness to conv2d, reduce_sum, and a Forge-generated workload set.",
  },
  {
    title: "Feature families beyond the corpus",
    detail: "cp.async bulk loads, TMA async copy, cluster/CGA primitives, and non-zero-accumulator tensor ops.",
  },
  {
    title: "Warp-reduce atomic lowering",
    detail: "Collapse per-lane atomics on a shared address into a REDUX → REDG warp-level path, matching ptxas optimization.",
  },
];

// ---------- hero stats ----------
const HERO_STATS = [
  {
    value: "142 / 142",
    label: "Corpus green",
    detail: "workbench + workbench_expanded, SM_120 verified",
  },
  {
    value: "bit-identical",
    label: "SAXPY correctness",
    detail: "output byte-for-byte equal to ptxas",
  },
  {
    value: "≈ 1.7×",
    label: "SAXPY throughput vs ptxas",
    detail: "same instructions, same GPRs, faster execution",
  },
];

// ---------- SAXPY PTX (actual kernel from benchmarks/saxpy_vs_nvidia.py) ----------
const SAXPY_PTX = `.version 9.0
.target sm_120
.address_size 64

.visible .entry saxpy(
    .param .u64 p_a,
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u32 n)
{
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r3;
    shl.b64 %rd1, %rd0, 2;
    ld.param.u64 %rd2, [p_a];
    ld.param.u64 %rd3, [p_x];
    ld.param.u64 %rd4, [p_y];
    add.s64 %rd5, %rd3, %rd1;
    add.s64 %rd6, %rd4, %rd1;

    ld.global.f32 %f1, [%rd2];
    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f3, [%rd6];
    fma.rn.f32 %f4, %f1, %f2, %f3;
    st.global.f32 [%rd6], %f4;
DONE:
    ret;
}`;

// Tiny PTX syntax colorer — keywords / types / registers / numbers / comments.
const colorPTX = (src) => {
  const KEYWORDS = new Set([
    "mov", "ld", "st", "mad", "setp", "bra", "cvt", "shl", "add", "fma",
    "ret", "entry", "param", "visible", "version", "target", "address_size",
    "global", "reg",
  ]);
  const TYPES = new Set([
    "u32", "u64", "s32", "s64", "b32", "b64", "f32", "f64",
    "pred", "lo", "rn", "ge",
  ]);
  const tokens = [];
  const re = /(\/\/[^\n]*|\.\w+|%\w+|0x[0-9a-fA-F]+|\d+|\w+|\s+|[^\w\s])/g;
  let m;
  while ((m = re.exec(src)) !== null) {
    const t = m[0];
    let cls = "text-slate-300";
    if (t.startsWith("//")) cls = "text-slate-500 italic";
    else if (t.startsWith("%")) cls = "text-violet-300";
    else if (t.startsWith(".")) {
      const core = t.slice(1);
      if (TYPES.has(core)) cls = "text-emerald-300";
      else cls = "text-cyan-400";
    } else if (/^0x[0-9a-fA-F]+$/.test(t) || /^\d+$/.test(t))
      cls = "text-amber-300";
    else if (KEYWORDS.has(t)) cls = "text-cyan-400 font-medium";
    else if (TYPES.has(t)) cls = "text-emerald-300";
    tokens.push({ t, cls });
  }
  return tokens;
};

// ---------- component ----------
export default function OpenPTXasShowcase() {
  const ptxTokens = React.useMemo(() => colorPTX(SAXPY_PTX), []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 antialiased selection:bg-cyan-400/30">
      {/* subtle animated background accent */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-[700px] h-[700px] rounded-full bg-cyan-500/10 blur-3xl" />
        <div className="absolute top-[40%] -left-40 w-[600px] h-[600px] rounded-full bg-violet-500/10 blur-3xl" />
      </div>

      {/* ---------- top nav ---------- */}
      <nav className="sticky top-0 z-50 backdrop-blur-md bg-slate-950/70 border-b border-slate-900">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2 font-mono text-sm">
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            <span className="text-white font-semibold">OpenPTXas</span>
            <span className="text-slate-500">· SM_120</span>
          </div>
          <div className="hidden md:flex items-center gap-6 text-sm text-slate-400">
            <a href="#stack" className="hover:text-white transition">Stack</a>
            <a href="#saxpy" className="hover:text-white transition">SAXPY</a>
            <a href="#corpus" className="hover:text-white transition">Corpus</a>
            <a href="#timeline" className="hover:text-white transition">Timeline</a>
            <a href="#next" className="hover:text-white transition">Next</a>
          </div>
          <div className="font-mono text-xs text-slate-500 hidden sm:block">
            HEAD <span className="text-cyan-400">033f398</span>
          </div>
        </div>
      </nav>

      {/* ---------- HERO ---------- */}
      <Section id="hero" className="pt-24 md:pt-32">
        <motion.div variants={fadeUp} className="flex justify-center mb-6">
          <span className="inline-flex items-center gap-2 text-xs font-mono tracking-wider uppercase text-cyan-400 bg-cyan-400/10 border border-cyan-400/20 px-3 py-1.5 rounded-full">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            corpus frozen · 142 / 142
          </span>
        </motion.div>

        <motion.h1
          variants={fadeUp}
          className="text-5xl md:text-7xl font-bold text-center tracking-tight leading-[1.05] mb-6"
        >
          <span className="bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
            OpenPTXas on SM_120
          </span>
          <br />
          <span className="bg-gradient-to-r from-cyan-400 via-cyan-300 to-violet-400 bg-clip-text text-transparent">
            Proven. Verified. Faster.
          </span>
        </motion.h1>

        <motion.p
          variants={fadeUp}
          className="text-lg md:text-xl text-slate-400 text-center max-w-2xl mx-auto mb-16 leading-relaxed"
        >
          A native PTX → SASS assembler for RTX 5090. 142/142 kernel corpus
          green. Bit-identical correctness. Measured execution wins over
          NVIDIA's ptxas.
        </motion.p>

        <div className="grid md:grid-cols-3 gap-4">
          {HERO_STATS.map((s) => (
            <Card key={s.label} className="text-center">
              <div className="text-3xl md:text-4xl font-bold bg-gradient-to-br from-cyan-300 to-violet-400 bg-clip-text text-transparent mb-2 font-mono">
                {s.value}
              </div>
              <div className="text-sm font-semibold text-white mb-1">{s.label}</div>
              <div className="text-xs text-slate-500">{s.detail}</div>
            </Card>
          ))}
        </div>
      </Section>

      {/* ---------- STACK ---------- */}
      <Section id="stack">
        <SectionHeader
          eyebrow="the stack"
          title="Forge → OpenCUDA → OpenPTXas → GPU"
          description="A fully-owned compilation path from a formally-verified source language down to native GPU instructions."
        />

        <div className="space-y-3">
          {STACK.map((step, i) => (
            <React.Fragment key={step.name}>
              <Card
                className={
                  step.highlight
                    ? "border-cyan-400/40 shadow-cyan-500/10"
                    : ""
                }
              >
                <div className="flex items-start gap-6">
                  <div className="flex-shrink-0">
                    <div
                      className={`w-12 h-12 rounded-xl flex items-center justify-center font-mono font-bold text-sm ${
                        step.highlight
                          ? "bg-cyan-400/20 text-cyan-300 border border-cyan-400/30"
                          : "bg-slate-800 text-slate-400 border border-slate-700"
                      }`}
                    >
                      0{i + 1}
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 mb-1">
                      <h3 className="text-xl font-semibold text-white">
                        {step.name}
                      </h3>
                      <span className="text-sm text-cyan-400 font-mono">
                        {step.role}
                      </span>
                      {step.highlight && (
                        <span className="text-[10px] font-semibold tracking-widest uppercase text-cyan-400 bg-cyan-400/10 border border-cyan-400/20 rounded-full px-2 py-0.5">
                          this project
                        </span>
                      )}
                    </div>
                    <p className="text-slate-400 text-sm leading-relaxed">
                      {step.detail}
                    </p>
                  </div>
                </div>
              </Card>
              {i < STACK.length - 1 && (
                <motion.div
                  variants={fadeUp}
                  className="flex justify-center text-slate-700"
                >
                  <IconArrow className="w-4 h-4" />
                </motion.div>
              )}
            </React.Fragment>
          ))}
        </div>
      </Section>

      {/* ---------- SAXPY ---------- */}
      <Section id="saxpy">
        <SectionHeader
          eyebrow="golden child"
          title="SAXPY — the canonical GPU baseline"
          description="y = a · x + y. One FMA per element, bounded by memory bandwidth. Same PTX source, same kernel shape — faster execution."
        />

        <div className="grid md:grid-cols-5 gap-6">
          {/* code block */}
          <motion.div variants={fadeUp} className="md:col-span-3">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden shadow-lg shadow-slate-950/50">
              <div className="flex items-center justify-between px-5 py-3 border-b border-slate-800 bg-slate-950/50">
                <div className="flex items-center gap-2 font-mono text-xs text-slate-500">
                  <IconCode className="w-3.5 h-3.5" />
                  saxpy.ptx
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-slate-700" />
                  <span className="w-2 h-2 rounded-full bg-slate-700" />
                  <span className="w-2 h-2 rounded-full bg-slate-700" />
                </div>
              </div>
              <pre className="p-5 text-[13px] font-mono leading-relaxed overflow-x-auto">
                <code>
                  {ptxTokens.map((tok, i) => (
                    <span key={i} className={tok.cls}>
                      {tok.t}
                    </span>
                  ))}
                </code>
              </pre>
            </div>
          </motion.div>

          {/* comparison + highlight */}
          <motion.div
            variants={fadeUp}
            className="md:col-span-2 flex flex-col gap-6"
          >
            <Card>
              <h3 className="text-sm font-semibold text-cyan-400 tracking-wider uppercase mb-4">
                ptxas vs OpenPTXas
              </h3>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs text-slate-500 uppercase tracking-wider">
                    <th className="text-left pb-3 font-medium">Metric</th>
                    <th className="text-right pb-3 font-medium">ptxas</th>
                    <th className="text-right pb-3 font-medium">Ours</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {[
                    ["Correctness", "✓", "✓", "match"],
                    ["Instructions", "23", "23", "match"],
                    ["GPR usage", "23", "23", "match"],
                    ["Text size", "512 B", "512 B", "match"],
                    ["Runtime (median)", "203 µs", "117 µs", "win"],
                    ["Bandwidth", "990 GB/s", "1716 GB/s", "win"],
                  ].map(([k, a, b, status]) => (
                    <tr key={k} className="text-slate-300">
                      <td className="py-3">{k}</td>
                      <td className="py-3 text-right font-mono text-slate-400">{a}</td>
                      <td
                        className={`py-3 text-right font-mono ${
                          status === "win"
                            ? "text-cyan-300 font-semibold"
                            : "text-slate-300"
                        }`}
                      >
                        {b}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            <Card className="bg-gradient-to-br from-cyan-500/10 to-violet-500/10 border-cyan-400/30">
              <div className="flex items-center gap-3 mb-2">
                <IconLightning className="w-5 h-5 text-cyan-400" />
                <div className="text-xs tracking-widest uppercase text-cyan-400 font-semibold">
                  Same kernel. Faster.
                </div>
              </div>
              <p className="text-white text-lg font-semibold leading-snug">
                Same PTX input. Same instruction count. Same register pressure.{" "}
                <span className="text-cyan-300">
                  Different scoreboard choices
                </span>{" "}
                → ~1.73× throughput.
              </p>
            </Card>
          </motion.div>
        </div>
      </Section>

      {/* ---------- CORPUS ---------- */}
      <Section id="corpus">
        <SectionHeader
          eyebrow="milestone"
          title="142 / 142 — Full Corpus Green"
          description="Every fixture in workbench.py + workbench_expanded.py compiles through OpenPTXas and executes correctly on hardware."
        />

        <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-3 mb-10">
          {CORPUS_CATEGORIES.map((c) => (
            <Card key={c.name} className="p-4">
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-cyan-400/15 border border-cyan-400/30 flex items-center justify-center text-cyan-300">
                  <IconCheck className="w-4 h-4" />
                </div>
                <div>
                  <div className="font-semibold text-white text-sm">{c.name}</div>
                  <div className="text-xs text-slate-500 mt-0.5 leading-relaxed">
                    {c.note}
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>

        <motion.div variants={fadeUp}>
          <Card>
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div>
                <div className="text-xs text-cyan-400 tracking-widest uppercase font-semibold mb-1">
                  regression gate
                </div>
                <div className="text-white font-semibold">
                  Deterministic full-corpus sweep, CI-gatable
                </div>
                <div className="text-sm text-slate-400 mt-1">
                  Fresh CUDA context per kernel. Exit 0 on green, 1 on any
                  failure.
                </div>
              </div>
              <div className="font-mono text-sm bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-slate-300 whitespace-nowrap">
                <span className="text-slate-600">$ </span>
                <span className="text-cyan-300">python</span>
                <span className="text-slate-200"> scripts/corpus_sweep.py</span>
              </div>
            </div>
          </Card>
        </motion.div>
      </Section>

      {/* ---------- TIMELINE ---------- */}
      <Section id="timeline">
        <SectionHeader
          eyebrow="campaign arc"
          title="R31 → R58"
          description="How the corpus went green. Each landing lands one proven root cause — nothing more."
        />

        <div className="relative">
          {/* vertical line */}
          <div
            className="absolute left-[19px] top-2 bottom-2 w-px bg-gradient-to-b from-cyan-400/40 via-slate-800 to-violet-400/40"
            aria-hidden
          />
          <div className="space-y-4">
            {TIMELINE.map((t) => (
              <motion.div
                key={t.tag}
                variants={fadeUp}
                className="relative pl-16"
              >
                <div
                  className={`absolute left-0 top-5 w-10 h-10 rounded-full flex items-center justify-center font-mono text-[11px] font-bold tracking-tight border-2 ${
                    t.closing
                      ? "bg-cyan-400 text-slate-950 border-cyan-300 shadow-lg shadow-cyan-500/30"
                      : t.big
                      ? "bg-slate-900 text-cyan-300 border-cyan-400/60"
                      : "bg-slate-900 text-slate-400 border-slate-700"
                  }`}
                >
                  {t.tag}
                </div>
                <Card
                  className={
                    t.big
                      ? "border-cyan-400/30"
                      : t.closing
                      ? "border-cyan-400/40 bg-cyan-500/5"
                      : ""
                  }
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h3 className="text-white font-semibold mb-1.5">
                        {t.title}
                      </h3>
                      <p className="text-sm text-slate-400 leading-relaxed">
                        {t.detail}
                      </p>
                    </div>
                    {t.closing && (
                      <span className="flex-shrink-0 text-[10px] font-semibold tracking-widest uppercase text-cyan-400 bg-cyan-400/10 border border-cyan-400/20 rounded-full px-2 py-1 whitespace-nowrap">
                        closing commit
                      </span>
                    )}
                    {t.big && !t.closing && (
                      <span className="flex-shrink-0 text-[10px] font-semibold tracking-widest uppercase text-violet-300 bg-violet-400/10 border border-violet-400/20 rounded-full px-2 py-1 whitespace-nowrap">
                        keystone
                      </span>
                    )}
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </Section>

      {/* ---------- PRINCIPLES ---------- */}
      <Section id="principles">
        <SectionHeader
          eyebrow="how it stays green"
          title="Design principles"
          description="A compiler's reputation is built over years and lost in a single bad landing. These principles are non-negotiable."
        />

        <div className="grid md:grid-cols-2 gap-4">
          {PRINCIPLES.map((p) => (
            <Card key={p.title}>
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-violet-500/15 border border-violet-400/25 text-violet-300 flex items-center justify-center">
                  <IconShield className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-white font-semibold mb-1.5">{p.title}</h3>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    {p.detail}
                  </p>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </Section>

      {/* ---------- NEXT PHASE ---------- */}
      <Section id="next">
        <SectionHeader
          eyebrow="what's next"
          title="Next phase"
          description="Correctness is frozen. The next arc is perf parity, CI hardening, and feature expansion."
        />

        <div className="grid md:grid-cols-2 gap-4">
          {NEXT_PHASE.map((n, i) => (
            <Card key={n.title}>
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-slate-800 border border-slate-700 text-slate-400 flex items-center justify-center font-mono text-sm">
                  0{i + 1}
                </div>
                <div>
                  <h3 className="text-white font-semibold mb-1.5">{n.title}</h3>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    {n.detail}
                  </p>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </Section>

      {/* ---------- FOOTER ---------- */}
      <footer className="border-t border-slate-900 bg-slate-950/80">
        <div className="max-w-6xl mx-auto px-6 py-8 flex flex-col md:flex-row items-center justify-between gap-4 text-sm">
          <div className="flex items-center gap-2 text-slate-500 font-mono">
            <span className="w-2 h-2 rounded-full bg-cyan-400/70" />
            OpenPTXas — native PTX → SASS for SM_120
          </div>
          <div className="text-slate-600 font-mono text-xs">
            corpus frozen · HEAD{" "}
            <span className="text-slate-400">033f398</span> · campaign R31 – R58
          </div>
        </div>
      </footer>
    </div>
  );
}
