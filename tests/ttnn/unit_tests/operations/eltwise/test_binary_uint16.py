# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


def compare_tensor_bits(t1: torch.Tensor, t2: torch.Tensor, max_print: int = 32):
    """
    Prints the values and binary patterns of two int32 tensors side by side.
    If there is a mismatch, the exact conflicting bits are highlighted in RED.

    Assumes:
    - both tensors are int32
    - both tensors have same shape
    """

    # 1. Defensive checks
    if t1.shape != t2.shape:
        raise ValueError("Tensors must have the same shape")

    if t1.dtype != torch.int32 or t2.dtype != torch.int32:
        raise TypeError("Both tensors must be int32")

    # 2. Performance booster: Extract entire tensors to Python lists at once
    flat1 = t1.flatten().detach().cpu().tolist()
    flat2 = t2.flatten().detach().cpu().tolist()

    # ANSI escape codes for terminal coloring
    RED = "\033[91m"
    RESET = "\033[0m"

    for i, (a_int, b_int) in enumerate(zip(flat1, flat2)):
        # Safety valve for console flooding
        if i >= max_print:
            print(f"--- Showing first {max_print} elements. {len(flat1) - max_print} elements omitted. ---")
            break

        # Convert to 32-bit binary strings (2's complement handling via & 0xFFFFFFFF)
        a_bits_raw = format(a_int & 0xFFFFFFFF, "032b")
        b_bits_raw = format(b_int & 0xFFFFFFFF, "032b")

        # OPTIMIZATION: Check the original numbers first
        if a_int == b_int:
            # --- FAST PATH: Exact match ---
            # Just chunk the bits into groups of 4 for clean reading
            bits_formatted = " ".join([a_bits_raw[j : j + 4] for j in range(0, 32, 4)])

            print(f"Index {i}: [MATCH]")
            print(f"  tensor1 value : {a_int}")
            print(f"  tensor1 bits  : {bits_formatted}")
            print(f"  tensor2 bits  : {bits_formatted}\n")

        else:
            # --- SLOW PATH: Mismatch detected ---
            # Inspect character by character to inject colors and a caret map
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
            print(f"  tensor1 value : {a_int}")
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
    print("is_equal", is_equal)
    compare_tensor_bits(a, out)
    assert is_equal, "Assert against golden failed"
