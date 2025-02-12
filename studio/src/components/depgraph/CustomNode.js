import React from 'react';
import { Handle, Position } from 'reactflow';

const CustomNode = ({ data }) => {
  return (
    <div className="custom-node bg-gray-800 border border-gray-700 rounded-lg p-4 shadow-lg">
      <Handle type="target" position={Position.Top} />
      <div className="text-white">
        {data.label}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};

export default CustomNode; 