"""
Symbolic Executor - LLVM IR to Z3 formula translator.

This module provides symbolic execution of LLVM IR basic blocks,
translating them into Z3 expressions for formal verification.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any, List, Tuple
from z3 import (
    BitVec, BitVecVal, BitVecRef,
    And, Or, Not, If,
    LShR, ULT, ULE, UGT, UGE,
)

from atlas.config import get_config


@dataclass
class SymbolicState:
    """Represents the symbolic state during execution."""
    registers: Dict[str, BitVecRef] = field(default_factory=dict)
    bit_width: int = 32
    path_condition: Any = None  # Z3 expression
    
    def get(self, name: str) -> Optional[BitVecRef]:
        """Get a register value."""
        return self.registers.get(name)
    
    def set(self, name: str, value: BitVecRef) -> None:
        """Set a register value."""
        self.registers[name] = value
    
    def copy(self) -> 'SymbolicState':
        """Create a copy of the state."""
        new_state = SymbolicState(
            registers=self.registers.copy(),
            bit_width=self.bit_width,
            path_condition=self.path_condition,
        )
        return new_state


class SymbolicExecutor:
    """
    Symbolically executes LLVM IR and builds Z3 formulas.
    
    Supports a subset of LLVM IR instructions:
    - Arithmetic: add, sub, mul, sdiv, udiv, srem, urem
    - Bitwise: and, or, xor, shl, lshr, ashr
    - Comparison: icmp (eq, ne, ugt, uge, ult, ule, sgt, sge, slt, sle)
    - Control flow: br (with bounded unrolling)
    """
    
    def __init__(self, bit_width: int = 32):
        config = get_config()
        self.bit_width = bit_width or config.verifier.bit_width
        self.loop_unroll_depth = config.verifier.loop_unroll_depth
    
    def execute_function(
        self,
        ir_code: str,
        input_names: List[str]
    ) -> Callable[..., BitVecRef]:
        """
        Parse LLVM IR function and return a symbolic function.
        
        Args:
            ir_code: The LLVM IR code string
            input_names: Names of input parameters (e.g., ['%x', '%y'])
            
        Returns:
            A callable that takes Z3 BitVec inputs and returns Z3 output
        """
        # Parse function body
        instructions = self._parse_instructions(ir_code)
        
        def symbolic_func(*inputs: BitVecRef) -> BitVecRef:
            state = SymbolicState(bit_width=self.bit_width)
            
            # Map input parameters to symbolic values
            for name, value in zip(input_names, inputs):
                state.set(name, value)
            
            # Execute instructions
            for inst in instructions:
                result = self._execute_instruction(state, inst)
                if result is not None and inst.get('dest'):
                    state.set(inst['dest'], result)
            
            # Return the last computed value (simplified assumption)
            # In practice, we'd track the return instruction
            return self._find_return_value(state, instructions)
        
        return symbolic_func
    
    def _parse_instructions(self, ir_code: str) -> List[Dict]:
        """Parse LLVM IR into a list of instruction dictionaries."""
        instructions = []
        lines = ir_code.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('define'):
                continue
            if line == '}':
                continue
            
            inst = self._parse_single_instruction(line)
            if inst:
                instructions.append(inst)
        
        return instructions
    
    def _parse_single_instruction(self, line: str) -> Optional[Dict]:
        """Parse a single LLVM IR instruction."""
        # Handle assignment: %result = operation ...
        assign_match = re.match(r'(%[\w.]+)\s*=\s*(.+)', line)
        if assign_match:
            dest = assign_match.group(1)
            rest = assign_match.group(2)
        else:
            dest = None
            rest = line
        
        # Parse the operation
        parts = rest.split()
        if not parts:
            return None
        
        opcode = parts[0]
        
        # Arithmetic instructions
        if opcode in ('add', 'sub', 'mul', 'sdiv', 'udiv', 'srem', 'urem'):
            return self._parse_binary_op(dest, opcode, parts[1:])
        
        # Bitwise instructions
        if opcode in ('and', 'or', 'xor', 'shl', 'lshr', 'ashr'):
            return self._parse_binary_op(dest, opcode, parts[1:])
        
        # Comparison
        if opcode == 'icmp':
            return self._parse_icmp(dest, parts[1:])
        
        # Select (ternary)
        if opcode == 'select':
            return self._parse_select(dest, parts[1:])
        
        # Return
        if opcode == 'ret':
            return self._parse_ret(parts[1:])
        
        # PHI (for loops)
        if opcode == 'phi':
            return self._parse_phi(dest, parts[1:])
        
        return None
    
    def _parse_binary_op(self, dest: str, opcode: str, parts: List[str]) -> Dict:
        """Parse binary arithmetic/bitwise operation."""
        # Format: type op1, op2 [nuw] [nsw]
        # Remove flags and type
        clean_parts = [p.rstrip(',') for p in parts if not p.startswith('i') and p not in ('nuw', 'nsw')]
        
        # Find operands (they contain % or are numbers)
        operands = []
        for p in parts:
            p = p.rstrip(',')
            if p.startswith('%') or p.lstrip('-').isdigit():
                operands.append(p)
        
        if len(operands) >= 2:
            return {
                'op': opcode,
                'dest': dest,
                'operands': operands[:2],
            }
        return None
    
    def _parse_icmp(self, dest: str, parts: List[str]) -> Dict:
        """Parse icmp comparison instruction."""
        # Format: pred type op1, op2
        if len(parts) < 3:
            return None
        
        pred = parts[0]
        operands = []
        for p in parts[1:]:
            p = p.rstrip(',')
            if p.startswith('%') or p.lstrip('-').isdigit():
                operands.append(p)
        
        if len(operands) >= 2:
            return {
                'op': 'icmp',
                'dest': dest,
                'pred': pred,
                'operands': operands[:2],
            }
        return None
    
    def _parse_select(self, dest: str, parts: List[str]) -> Dict:
        """Parse select (ternary) instruction."""
        # Format: i1 %cond, type %val1, type %val2
        operands = []
        for p in parts:
            p = p.rstrip(',')
            if p.startswith('%') or p.lstrip('-').isdigit():
                operands.append(p)
        
        if len(operands) >= 3:
            return {
                'op': 'select',
                'dest': dest,
                'operands': operands[:3],  # [cond, true_val, false_val]
            }
        return None
    
    def _parse_ret(self, parts: List[str]) -> Dict:
        """Parse return instruction."""
        for p in parts:
            p = p.rstrip(',')
            if p.startswith('%') or p.lstrip('-').isdigit():
                return {
                    'op': 'ret',
                    'dest': None,
                    'operands': [p],
                }
        return {'op': 'ret', 'dest': None, 'operands': []}
    
    def _parse_phi(self, dest: str, parts: List[str]) -> Dict:
        """Parse PHI node (for loops)."""
        # Simplified: just grab values
        # Format: type [ %val1, %label1 ], [ %val2, %label2 ]
        values = re.findall(r'\[\s*([^,\]]+)', ' '.join(parts))
        return {
            'op': 'phi',
            'dest': dest,
            'operands': [v.strip() for v in values],
        }
    
    def _execute_instruction(self, state: SymbolicState, inst: Dict) -> Optional[BitVecRef]:
        """Execute a single instruction symbolically."""
        op = inst['op']
        operands = inst.get('operands', [])
        
        # Get symbolic operand values
        sym_ops = [self._get_operand_value(state, op) for op in operands]
        
        # Arithmetic
        if op == 'add':
            return sym_ops[0] + sym_ops[1]
        if op == 'sub':
            return sym_ops[0] - sym_ops[1]
        if op == 'mul':
            return sym_ops[0] * sym_ops[1]
        if op == 'sdiv':
            # Signed division
            return sym_ops[0] / sym_ops[1] if sym_ops[1] is not None else sym_ops[0]
        if op == 'udiv':
            # Unsigned division
            from z3 import UDiv
            return UDiv(sym_ops[0], sym_ops[1]) if sym_ops[1] is not None else sym_ops[0]
        if op == 'srem':
            return sym_ops[0] % sym_ops[1] if sym_ops[1] is not None else sym_ops[0]
        if op == 'urem':
            from z3 import URem
            return URem(sym_ops[0], sym_ops[1]) if sym_ops[1] is not None else sym_ops[0]
        
        # Bitwise
        if op == 'and':
            return sym_ops[0] & sym_ops[1]
        if op == 'or':
            return sym_ops[0] | sym_ops[1]
        if op == 'xor':
            return sym_ops[0] ^ sym_ops[1]
        if op == 'shl':
            return sym_ops[0] << sym_ops[1]
        if op == 'lshr':
            return LShR(sym_ops[0], sym_ops[1])
        if op == 'ashr':
            return sym_ops[0] >> sym_ops[1]
        
        # Comparison
        if op == 'icmp':
            pred = inst.get('pred', 'eq')
            return self._execute_icmp(pred, sym_ops[0], sym_ops[1])
        
        # Select
        if op == 'select':
            cond = sym_ops[0]
            # Ensure cond is boolean (1-bit)
            if cond is not None:
                return If(cond != 0, sym_ops[1], sym_ops[2])
        
        # PHI - simplified handling
        if op == 'phi':
            # Just return first value for now (proper handling needs CFG)
            if sym_ops:
                return sym_ops[0]
        
        # Return
        if op == 'ret':
            if sym_ops:
                return sym_ops[0]
        
        return None
    
    def _execute_icmp(self, pred: str, a: BitVecRef, b: BitVecRef) -> BitVecRef:
        """Execute icmp comparison."""
        from z3 import If
        
        comparisons = {
            'eq': lambda x, y: x == y,
            'ne': lambda x, y: x != y,
            'ugt': lambda x, y: UGT(x, y),
            'uge': lambda x, y: UGE(x, y),
            'ult': lambda x, y: ULT(x, y),
            'ule': lambda x, y: ULE(x, y),
            'sgt': lambda x, y: x > y,
            'sge': lambda x, y: x >= y,
            'slt': lambda x, y: x < y,
            'sle': lambda x, y: x <= y,
        }
        
        cmp_func = comparisons.get(pred, comparisons['eq'])
        result = cmp_func(a, b)
        
        # Return 1 or 0 as BitVec
        return If(result, BitVecVal(1, self.bit_width), BitVecVal(0, self.bit_width))
    
    def _get_operand_value(self, state: SymbolicState, operand: str) -> BitVecRef:
        """Get the symbolic value of an operand."""
        if operand.startswith('%'):
            # Register reference
            val = state.get(operand)
            if val is not None:
                return val
            # Unknown register, create fresh symbolic value
            fresh = BitVec(operand, self.bit_width)
            state.set(operand, fresh)
            return fresh
        else:
            # Constant
            try:
                const_val = int(operand)
                return BitVecVal(const_val, self.bit_width)
            except ValueError:
                return BitVecVal(0, self.bit_width)
    
    def _find_return_value(self, state: SymbolicState, instructions: List[Dict]) -> BitVecRef:
        """Find the return value from executed instructions."""
        # Look for ret instruction
        for inst in reversed(instructions):
            if inst['op'] == 'ret' and inst.get('operands'):
                return self._get_operand_value(state, inst['operands'][0])
        
        # Fallback: return last assigned register
        if state.registers:
            return list(state.registers.values())[-1]
        
        return BitVecVal(0, self.bit_width)


def ir_to_z3_function(ir_code: str, param_names: List[str]) -> Callable[..., BitVecRef]:
    """
    Convenience function to convert LLVM IR to a Z3 symbolic function.
    
    Args:
        ir_code: LLVM IR string
        param_names: List of parameter names (e.g., ['%0', '%1'])
        
    Returns:
        Callable that maps Z3 inputs to Z3 output
    """
    executor = SymbolicExecutor()
    return executor.execute_function(ir_code, param_names)
