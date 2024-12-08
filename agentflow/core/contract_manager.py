"""Interface contract manager for AgentFlow."""
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from .exceptions import ContractViolationError

@dataclass
class ContractSpec:
    required_fields: Set[str]
    optional_fields: Set[str]

class ContractManager:
    """管理Agent接口契约的核心组件"""
    
    def __init__(self, contract_spec: Dict[str, Any]):
        self.input_contract = ContractSpec(
            required_fields=set(contract_spec.get("INPUT_CONTRACT", {}).get("REQUIRED_FIELDS", [])),
            optional_fields=set(contract_spec.get("INPUT_CONTRACT", {}).get("OPTIONAL_FIELDS", []))
        )
        
        self.output_contract = ContractSpec(
            required_fields=set(contract_spec.get("OUTPUT_CONTRACT", {}).get("MANDATORY_FIELDS", [])),
            optional_fields=set(contract_spec.get("OUTPUT_CONTRACT", {}).get("OPTIONAL_FIELDS", []))
        )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据是否符合契约
        
        Args:
            input_data: 输入数据
            
        Returns:
            验证是否通过
            
        Raises:
            ContractViolationError: 当输入数据违反契约时
        """
        return self._validate_against_contract(input_data, self.input_contract, "input")

    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """验证输出数据是否符合契约
        
        Args:
            output_data: 输出数据
            
        Returns:
            验证是否通过
            
        Raises:
            ContractViolationError: 当输出数据违反契约时
        """
        return self._validate_against_contract(output_data, self.output_contract, "output")

    def _validate_against_contract(
        self, 
        data: Dict[str, Any], 
        contract: ContractSpec,
        contract_type: str
    ) -> bool:
        """根据契约验证数据
        
        Args:
            data: 待验证数据
            contract: 契约规范
            contract_type: 契约类型(input/output)
            
        Returns:
            验证是否通过
            
        Raises:
            ContractViolationError: 当数据违反契约时
        """
        if not isinstance(data, dict):
            raise ContractViolationError(
                f"Invalid {contract_type} type: expected dict, got {type(data)}"
            )
            
        # 验证必需字段
        missing_fields = contract.required_fields - set(data.keys())
        if missing_fields:
            raise ContractViolationError(
                f"Missing required {contract_type} fields: {missing_fields}"
            )
            
        # 验证字段有效性
        invalid_fields = set(data.keys()) - (contract.required_fields | contract.optional_fields)
        if invalid_fields:
            raise ContractViolationError(
                f"Invalid {contract_type} fields: {invalid_fields}"
            )
            
        return True

    def get_input_schema(self) -> Dict[str, Any]:
        """获取输入数据模式"""
        return {
            "type": "object",
            "required": list(self.input_contract.required_fields),
            "properties": {
                field: {"type": "any"} 
                for field in (self.input_contract.required_fields | self.input_contract.optional_fields)
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """获取输出数据模式"""
        return {
            "type": "object",
            "required": list(self.output_contract.required_fields),
            "properties": {
                field: {"type": "any"}
                for field in (self.output_contract.required_fields | self.output_contract.optional_fields)
            }
        }

    def describe_contracts(self) -> Dict[str, Any]:
        """获取契约描述"""
        return {
            "input_contract": {
                "required_fields": list(self.input_contract.required_fields),
                "optional_fields": list(self.input_contract.optional_fields)
            },
            "output_contract": {
                "required_fields": list(self.output_contract.required_fields),
                "optional_fields": list(self.output_contract.optional_fields)
            }
        }
