"""Test that the PTX parser handles TMA/mbarrier syntax correctly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ptx.parser import _tokenize, _Parser


def _parse_stmt(text):
    """Parse a single PTX statement."""
    tokens = _tokenize(text)
    p = _Parser(tokens)
    return p._parse_statement()


def test_cp_async_bulk_shared_global():
    """cp.async.bulk.shared::cluster.global parses as op='cp' with correct types."""
    instr = _parse_stmt(
        'cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes '
        '[smem], [%rd0], %r0, [mbar];')
    assert instr.op == 'cp'
    assert 'async' in instr.types
    assert 'bulk' in instr.types
    # shared::cluster should be a single type entry
    assert any('shared' in t and 'cluster' in t for t in instr.types)


def test_cp_async_bulk_tensor_1d():
    """cp.async.bulk.tensor.1d parses correctly."""
    instr = _parse_stmt(
        'cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes '
        '[smem], [%rd0, {%r0}], [mbar];')
    assert instr.op == 'cp'
    assert 'tensor' in instr.types
    assert '1d' in instr.types


def test_cp_async_bulk_tensor_2d():
    """cp.async.bulk.tensor.2d parses correctly."""
    instr = _parse_stmt(
        'cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes '
        '[smem], [%rd0, {%r0, %r1}], [mbar];')
    assert instr.op == 'cp'
    assert 'tensor' in instr.types
    assert '2d' in instr.types


def test_mbarrier_init():
    """mbarrier.init.shared::cta.b64 is parsed as op='mbarrier'."""
    instr = _parse_stmt('mbarrier.init.shared::cta.b64 [mbar], 1;')
    assert instr.op == 'mbarrier'
    assert 'init' in instr.types
    assert any('shared' in t for t in instr.types)
    # dest should be None (mbarrier is void op)
    assert instr.dest is None


def test_mbarrier_arrive():
    """mbarrier.arrive parses correctly."""
    instr = _parse_stmt('mbarrier.arrive.shared::cta.b64 %rd0, [mbar];')
    assert instr.op == 'mbarrier'
    assert 'arrive' in instr.types


def test_mbarrier_try_wait():
    """mbarrier.try_wait.parity parses correctly."""
    instr = _parse_stmt(
        'mbarrier.try_wait.parity.shared::cta.b64 %p0, [mbar], 0;')
    assert instr.op == 'mbarrier'
    assert 'try_wait' in instr.types


def test_cp_async_bulk_commit_group():
    """cp.async.bulk.commit_group parses correctly."""
    instr = _parse_stmt('cp.async.bulk.commit_group;')
    assert instr.op == 'cp'
    assert 'bulk' in instr.types
    assert 'commit_group' in instr.types


def test_cp_async_bulk_wait_group():
    """cp.async.bulk.wait_group parses correctly."""
    instr = _parse_stmt('cp.async.bulk.wait_group 0;')
    assert instr.op == 'cp'
    assert 'bulk' in instr.types
    assert 'wait_group' in instr.types


def test_cp_async_bulk_store():
    """cp.async.bulk store direction parses correctly."""
    instr = _parse_stmt(
        'cp.async.bulk.global.shared::cta.bulk_group [%rd0], [smem], %r0;')
    assert instr.op == 'cp'
    assert 'bulk' in instr.types
    # global appears before shared in type list
    types_str = '.'.join(instr.types)
    assert 'global' in types_str


def test_double_colon_in_ident():
    """PTX namespace separator :: is preserved in type qualifiers."""
    instr = _parse_stmt('mbarrier.init.shared::cta.b64 [mbar], 1;')
    # shared::cta should be one qualifier, not split
    assert any('::' in t for t in instr.types), f"types={instr.types}"


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    for t in tests:
        try:
            t()
            print(f'  PASS  {t.__name__}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {t.__name__}: {e}')
    print(f'\n{passed}/{len(tests)} passed')
