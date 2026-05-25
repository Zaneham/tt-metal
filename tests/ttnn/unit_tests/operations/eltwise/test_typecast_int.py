# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import struct
import torch
import ttnn
import pytest


def compare_tensor_bits(t1: torch.Tensor, t2: torch.Tensor, max_print: int = 32):
    """Prints the values and binary patterns of two int32 or float32 tensors side by side.

    If there is a mismatch, the exact conflicting bits are highlighted in RED.

    Assumes:
    - both tensors are either int32 or float32 (and match each other)
    - both tensors have same shape
    """

    # 1. Defensive checks
    if t1.shape != t2.shape:
        raise ValueError("Tensors must have the same shape")

    if t1.dtype != t2.dtype:
        raise TypeError("Both tensors must have the exact same dtype")

    if t1.dtype not in (torch.int32, torch.float32):
        raise TypeError("Supported dtypes are torch.int32 or torch.float32")

    is_float = t1.dtype == torch.float32

    # 2. Performance booster: Extract entire tensors to Python lists at once
    flat1 = t1.flatten().detach().cpu().tolist()
    flat2 = t2.flatten().detach().cpu().tolist()

    # ANSI escape codes for terminal coloring
    RED = "\033[91m"
    RESET = "\033[0m"

    for i, (a_val, b_val) in enumerate(zip(flat1, flat2)):
        # Safety valve for console flooding
        if i >= max_print:
            print(f"--- Showing first {max_print} elements. {len(flat1) - max_print} elements omitted. ---")
            break

        # Convert to 32-bit binary strings
        if is_float:
            # IEEE 754 Bit Reinterpretation (float -> 4 bytes -> uint32 integer)
            a_int = struct.unpack("<I", struct.pack("<f", a_val))[0]
            b_int = struct.unpack("<I", struct.pack("<f", b_val))[0]
        else:
            # 2's complement handling for integers
            a_int = a_val & 0xFFFFFFFF
            b_int = b_val & 0xFFFFFFFF

        a_bits_raw = format(a_int, "032b")
        b_bits_raw = format(b_int, "032b")

        # Check for strict value match (handles NaNs safely via bit equality)
        if a_bits_raw == b_bits_raw:
            # --- FAST PATH: Exact match ---
            bits_formatted = " ".join([a_bits_raw[j : j + 4] for j in range(0, 32, 4)])

            print(f"Index {i}: [MATCH]")
            print(f"  tensor1 value : {a_val}")
            print(f"  tensor1 bits  : {bits_formatted}")
            print(f"  tensor2 bits  : {bits_formatted}\n")

        else:
            # --- SLOW PATH: Mismatch detected ---
            a_styled = []
            b_styled = []
            diff_marker = []

            for idx, (bit_a, bit_b) in enumerate(zip(a_bits_raw, b_bits_raw)):
                # Inject a spacing gap every 4 bits
                if idx > 0 and idx % 4 == 0:
                    a_styled.append(" ")
                    b_styled.append(" ")
                    diff_marker.append(" ")

                if bit_a != bit_b:
                    # Highlight discrepancies in Red
                    a_styled.append(f"{RED}{bit_a}{RESET}")
                    b_styled.append(f"{RED}{bit_b}{RESET}")
                    diff_marker.append("^")
                else:
                    a_styled.append(bit_a)
                    b_styled.append(bit_b)
                    diff_marker.append(".")

            print(f"Index {i}: [MISMATCH]")
            print(f"  tensor1 value : {a_val}")
            print(f"  tensor1 bits  : {''.join(a_styled)}")
            print(f"  tensor2 bits  : {''.join(b_styled)}")
            print(f"  bit mismatches: {''.join(diff_marker)}\n")


pytestmark = pytest.mark.use_module_device


def test_typecast_1tile(device):
    a = torch.linspace(0, 65535, steps=1024, dtype=torch.int32)
    ta = ttnn.from_torch(a, dtype=ttnn.uint16, device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.typecast(ta, dtype=ttnn.uint32)
    out = ttnn.to_torch(result, dtype=torch.int32)
    is_equal = torch.equal(a, out)
    compare_tensor_bits(a, out)
    print("is_equal", is_equal)
    assert is_equal, "Assert against golden failed"


def test_typecast_1tile_fp(device):
    a = torch.linspace(0, 65535, steps=1024, dtype=torch.int32)
    float_a = a.to(torch.float32)
    ta = ttnn.from_torch(a, dtype=ttnn.uint16, device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.typecast(ta, dtype=ttnn.float32)
    out = ttnn.to_torch(result, dtype=torch.float32)
    is_equal = torch.equal(a, out)
    compare_tensor_bits(float_a, out)
    print("is_equal", is_equal)
    assert is_equal, "Assert against golden failed"


# use case for TG Llama : need to achieve (int32 + int32) addition with (uint16 + int32) inputs
def test_typecast_uint16(device):
    torch.manual_seed(0)

    in_data1 = torch.tensor([[[[700, 100, 65000, 9500]]]], dtype=torch.int32)
    in_data2 = torch.tensor([[[[70000, 1000, 65000, 95000]]]], dtype=torch.int32)

    input_mem_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )
    input_tensor2 = ttnn.from_torch(
        in_data2,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    input_tensor3 = ttnn.typecast(
        input_tensor1,
        ttnn.uint32,
        memory_config=input_mem_config,
    )

    input_tensor3 = ttnn.typecast(
        input_tensor3,
        ttnn.int32,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.add(input_tensor3, input_tensor2)

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)
    golden_function = ttnn.get_golden_function(ttnn.add)
    golden_tensor = golden_function(in_data1, in_data2)

    assert torch.equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "shape, sub_core_grid",
    [
        (
            (torch.Size([1, 2, 32, 960])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 7, 32, 96])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 8, 32, 128])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 17, 32, 32])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
    ],
)
def test_typecast_subcore_grid(device, shape, sub_core_grid):
    torch.manual_seed(0)

    in_data1 = torch.randint(0, 65500, (shape), dtype=torch.int32)
    in_data2 = torch.randint(0, 128000, (shape), dtype=torch.int32)

    input_mem_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )
    input_tensor2 = ttnn.from_torch(
        in_data2,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    input_tensor3 = ttnn.typecast(
        input_tensor1,
        ttnn.uint32,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )

    input_tensor3 = ttnn.typecast(
        input_tensor3,
        ttnn.int32,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )

    output_tensor = ttnn.add(input_tensor3, input_tensor2)

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)
    golden_function = ttnn.get_golden_function(ttnn.add)
    golden_tensor = golden_function(in_data1, in_data2)

    assert torch.equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "shape, sub_core_grid",
    [
        # Large tensors: many tiles per core stresses the CB allocation.
        # Before the fix, TypecastSubgridProgramFactory allocated all per-core tiles
        # into CBs at once (ntiles_per_block * 2), overflowing L1 for large tensors.
        (
            torch.Size([1, 1, 1024, 2048]),  # 2048 tiles, ~1024 tiles/core with 2 cores
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))]),
        ),
        (
            torch.Size([1, 2, 2048, 2048]),  # 8192 tiles across 7 cores
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 6))]),
        ),
        (
            torch.Size([1, 4, 1024, 2048]),  # 8192 tiles across a multi-range sub_core_grid
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
    ],
)
def test_typecast_subcore_grid_large_tensor(device, shape, sub_core_grid):
    """Regression test: large tensors with sub_core_grids must not overflow L1."""
    torch.manual_seed(0)

    in_data = torch.randint(0, 65500, shape, dtype=torch.int32)
    input_mem_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.typecast(
        input_tensor,
        ttnn.uint32,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )

    output_tensor = ttnn.typecast(
        output_tensor,
        ttnn.int32,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )

    result = ttnn.to_torch(output_tensor, dtype=torch.int32)
    assert torch.equal(in_data, result)


@pytest.mark.parametrize(
    "shape, output_dtype, sub_core_grid",
    [
        (
            torch.Size([1, 1, 1024, 2048]),
            ttnn.bfloat8_b,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 6))]),
        ),
        (
            torch.Size([1, 1, 1024, 2048]),
            ttnn.bfloat4_b,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 6))]),
        ),
        (
            torch.Size([1, 4, 2048, 2048]),
            ttnn.bfloat8_b,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
        (
            torch.Size([1, 4, 2048, 2048]),
            ttnn.bfloat4_b,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
    ],
)
def test_typecast_bfloat_subcore_grid_large_tensor(device, shape, output_dtype, sub_core_grid):
    """Regression test: bfloat16 -> bfloat8_b/bfloat4_b typecast with sub_core_grids on large tensors."""
    torch.manual_seed(0)

    in_data = torch.randn(shape, dtype=torch.bfloat16)
    input_mem_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.typecast(
        input_tensor,
        output_dtype,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )
    assert output_tensor.dtype == output_dtype
    assert list(output_tensor.shape) == list(shape)

    roundtrip = ttnn.typecast(
        output_tensor,
        ttnn.bfloat16,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )
    result = ttnn.to_torch(roundtrip).to(torch.float32)
    reference = in_data.to(torch.float32)

    pcc = torch.corrcoef(torch.stack([reference.flatten(), result.flatten()]))[0, 1].item()
    min_pcc = 0.95 if output_dtype == ttnn.bfloat4_b else 0.99
    assert pcc >= min_pcc, f"PCC {pcc:.4f} below threshold {min_pcc} for {output_dtype}"


# for range verification in conversions
def test_typecast_uint16_subcore_grid(device):
    in_data1 = torch.tensor([[[[700, 100, 65000, 9500]]]], dtype=torch.int32)
    in_data2 = torch.tensor([[[[70000, 1000, 65000, 95000]]]], dtype=torch.int32)

    input_mem_config = ttnn.DRAM_MEMORY_CONFIG
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
        ]
    )

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )
    input_tensor2 = ttnn.from_torch(
        in_data2,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    input_tensor3 = ttnn.typecast(
        input_tensor1,
        ttnn.uint32,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grids,
    )
    input_tensor3 = ttnn.typecast(
        input_tensor3,
        ttnn.int32,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grids,
    )
    output_tensor = ttnn.add(input_tensor3, input_tensor2)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)
    golden_function = ttnn.get_golden_function(ttnn.add)
    golden_tensor = golden_function(in_data1, in_data2)

    assert torch.equal(golden_tensor, output_tensor)
