`<template>
  <div class="workflow-editor">
    <!-- Tool Panel -->
    <div class="tool-panel">
      <div class="tool-section">
        <h3>Agents</h3>
        <div v-for="agent in agentNodes" 
             :key="agent.type"
             class="tool-item"
             draggable="true"
             @dragstart="onDragStart($event, agent)">
          {{ agent.label }}
        </div>
      </div>
      
      <div class="tool-section">
        <h3>Processors</h3>
        <div v-for="processor in processorNodes"
             :key="processor.type"
             class="tool-item"
             draggable="true"
             @dragstart="onDragStart($event, processor)">
          {{ processor.label }}
        </div>
      </div>
      
      <div class="tool-section">
        <h3>Input/Output</h3>
        <div v-for="io in ioNodes"
             :key="io.type"
             class="tool-item"
             draggable="true"
             @dragstart="onDragStart($event, io)">
          {{ io.label }}
        </div>
      </div>
    </div>

    <!-- Main Canvas -->
    <div class="canvas-container" 
         ref="canvas"
         @drop="onDrop"
         @dragover.prevent>
      <!-- Nodes -->
      <div v-for="node in nodes"
           :key="node.id"
           class="node"
           :class="node.type"
           :style="getNodeStyle(node)"
           @mousedown="startDragNode($event, node)">
        <!-- Node Header -->
        <div class="node-header">
          {{ node.label }}
          <button @click="deleteNode(node)" class="delete-btn">×</button>
        </div>
        
        <!-- Node Inputs -->
        <div class="node-inputs">
          <div v-for="input in node.inputs"
               :key="input.id"
               class="node-port input-port"
               @mousedown="startConnection($event, node, input, 'input')">
            {{ input.label }}
          </div>
        </div>
        
        <!-- Node Content -->
        <div class="node-content">
          <component :is="node.component"
                    v-if="node.component"
                    v-bind="node.props"
                    @update="updateNodeData(node, $event)"/>
        </div>
        
        <!-- Node Outputs -->
        <div class="node-outputs">
          <div v-for="output in node.outputs"
               :key="output.id"
               class="node-port output-port"
               @mousedown="startConnection($event, node, output, 'output')">
            {{ output.label }}
          </div>
        </div>
      </div>

      <!-- Connections -->
      <svg class="connections">
        <path v-for="conn in connections"
              :key="conn.id"
              :d="getConnectionPath(conn)"
              :class="['connection', conn.active ? 'active' : '']"/>
      </svg>

      <!-- Preview Connection -->
      <svg class="preview-connection" v-if="previewConnection">
        <path :d="previewConnection"
              class="connection preview"/>
      </svg>
    </div>

    <!-- Properties Panel -->
    <div class="properties-panel" v-if="selectedNode">
      <div class="panel-header">
        <h3>Properties</h3>
        <button @click="selectedNode = null">×</button>
      </div>
      
      <div class="panel-content">
        <component :is="selectedNode.propertiesComponent"
                  v-if="selectedNode.propertiesComponent"
                  :node="selectedNode"
                  @update="updateNodeProperties"/>
      </div>
    </div>
  </div>
</template>

<script>
import { v4 as uuidv4 } from 'uuid'
import { defineComponent, ref, computed } from 'vue'

// Node Components
import AgentNode from './nodes/AgentNode.vue'
import ProcessorNode from './nodes/ProcessorNode.vue'
import InputNode from './nodes/InputNode.vue'
import OutputNode from './nodes/OutputNode.vue'

// Node Definitions
const NODE_TYPES = {
  AGENT: {
    component: AgentNode,
    inputs: [
      { id: 'input', label: 'Input' }
    ],
    outputs: [
      { id: 'output', label: 'Output' }
    ]
  },
  PROCESSOR: {
    component: ProcessorNode,
    inputs: [
      { id: 'input', label: 'Input' }
    ],
    outputs: [
      { id: 'output', label: 'Output' }
    ]
  },
  INPUT: {
    component: InputNode,
    outputs: [
      { id: 'output', label: 'Output' }
    ]
  },
  OUTPUT: {
    component: OutputNode,
    inputs: [
      { id: 'input', label: 'Input' }
    ]
  }
}

export default defineComponent({
  name: 'WorkflowEditor',
  
  setup() {
    const nodes = ref([])
    const connections = ref([])
    const selectedNode = ref(null)
    const canvas = ref(null)
    
    // Dragging state
    const isDragging = ref(false)
    const dragStartPos = ref({ x: 0, y: 0 })
    const draggedNode = ref(null)
    
    // Connection state
    const isConnecting = ref(false)
    const connectionStart = ref(null)
    const previewConnection = ref(null)
    
    // Available nodes for toolbox
    const agentNodes = [
      { type: 'AGENT', label: 'LLM Agent' },
      { type: 'AGENT', label: 'Search Agent' },
      { type: 'AGENT', label: 'Tool Agent' }
    ]
    
    const processorNodes = [
      { type: 'PROCESSOR', label: 'Text Processor' },
      { type: 'PROCESSOR', label: 'Data Transformer' },
      { type: 'PROCESSOR', label: 'Filter' }
    ]
    
    const ioNodes = [
      { type: 'INPUT', label: 'Input' },
      { type: 'OUTPUT', label: 'Output' }
    ]
    
    // Methods
    const addNode = (type, position) => {
      const nodeType = NODE_TYPES[type]
      if (!nodeType) return
      
      const node = {
        id: uuidv4(),
        type,
        label: type,
        position,
        inputs: [...(nodeType.inputs || [])],
        outputs: [...(nodeType.outputs || [])],
        component: nodeType.component,
        data: {}
      }
      
      nodes.value.push(node)
      return node
    }
    
    const deleteNode = (node) => {
      // Remove connected connections
      connections.value = connections.value.filter(conn => 
        conn.from.node !== node.id && conn.to.node !== node.id
      )
      
      // Remove node
      const index = nodes.value.findIndex(n => n.id === node.id)
      if (index >= 0) {
        nodes.value.splice(index, 1)
      }
      
      if (selectedNode.value?.id === node.id) {
        selectedNode.value = null
      }
    }
    
    const startDragNode = (event, node) => {
      if (event.target.classList.contains('node-port')) return
      
      isDragging.value = true
      draggedNode.value = node
      dragStartPos.value = {
        x: event.clientX - node.position.x,
        y: event.clientY - node.position.y
      }
      
      document.addEventListener('mousemove', onDragMove)
      document.addEventListener('mouseup', stopDragNode)
    }
    
    const onDragMove = (event) => {
      if (!isDragging.value || !draggedNode.value) return
      
      draggedNode.value.position = {
        x: event.clientX - dragStartPos.value.x,
        y: event.clientY - dragStartPos.value.y
      }
    }
    
    const stopDragNode = () => {
      isDragging.value = false
      draggedNode.value = null
      
      document.removeEventListener('mousemove', onDragMove)
      document.removeEventListener('mouseup', stopDragNode)
    }
    
    const startConnection = (event, node, port, type) => {
      event.stopPropagation()
      
      isConnecting.value = true
      connectionStart.value = { node: node.id, port: port.id, type }
      
      document.addEventListener('mousemove', updatePreviewConnection)
      document.addEventListener('mouseup', stopConnection)
    }
    
    const updatePreviewConnection = (event) => {
      if (!isConnecting.value) return
      
      const canvasRect = canvas.value.getBoundingClientRect()
      const endPoint = {
        x: event.clientX - canvasRect.left,
        y: event.clientY - canvasRect.top
      }
      
      previewConnection.value = generateConnectionPath(
        getPortPosition(connectionStart.value),
        endPoint
      )
    }
    
    const stopConnection = (event) => {
      isConnecting.value = false
      previewConnection.value = null
      connectionStart.value = null
      
      document.removeEventListener('mousemove', updatePreviewConnection)
      document.removeEventListener('mouseup', stopConnection)
    }
    
    const addConnection = (from, to) => {
      // Check if connection already exists
      const exists = connections.value.some(conn =>
        conn.from.node === from.node &&
        conn.from.port === from.port &&
        conn.to.node === to.node &&
        conn.to.port === to.port
      )
      
      if (!exists) {
        connections.value.push({
          id: uuidv4(),
          from,
          to
        })
      }
    }
    
    const getPortPosition = (nodeId, portId, type) => {
      const node = nodes.value.find(n => n.id === nodeId)
      if (!node) return { x: 0, y: 0 }
      
      const nodeEl = document.querySelector(`#node-${nodeId}`)
      if (!nodeEl) return { x: 0, y: 0 }
      
      const portEl = nodeEl.querySelector(`#port-${portId}`)
      if (!portEl) return { x: 0, y: 0 }
      
      const nodeRect = nodeEl.getBoundingClientRect()
      const portRect = portEl.getBoundingClientRect()
      
      return {
        x: portRect.left + portRect.width / 2 - nodeRect.left,
        y: portRect.top + portRect.height / 2 - nodeRect.top
      }
    }
    
    const generateConnectionPath = (start, end) => {
      const dx = end.x - start.x
      const dy = end.y - start.y
      const curve = Math.min(Math.abs(dx) / 2, 50)
      
      return `M ${start.x} ${start.y} 
              C ${start.x + curve} ${start.y},
                ${end.x - curve} ${end.y},
                ${end.x} ${end.y}`
    }
    
    return {
      nodes,
      connections,
      selectedNode,
      canvas,
      agentNodes,
      processorNodes,
      ioNodes,
      previewConnection,
      addNode,
      deleteNode,
      startDragNode,
      startConnection,
      addConnection,
      getPortPosition,
      generateConnectionPath
    }
  }
})
</script>

<style scoped>
.workflow-editor {
  display: flex;
  height: 100%;
  background: #1e1e1e;
  color: #fff;
}

.tool-panel {
  width: 200px;
  padding: 1rem;
  background: #252526;
  border-right: 1px solid #333;
  overflow-y: auto;
}

.tool-section {
  margin-bottom: 1rem;
}

.tool-section h3 {
  color: #ccc;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}

.tool-item {
  padding: 0.5rem;
  margin-bottom: 0.5rem;
  background: #333;
  border-radius: 4px;
  cursor: move;
  user-select: none;
}

.tool-item:hover {
  background: #444;
}

.canvas-container {
  flex: 1;
  position: relative;
  overflow: hidden;
}

.node {
  position: absolute;
  min-width: 150px;
  background: #2d2d2d;
  border: 1px solid #444;
  border-radius: 6px;
  user-select: none;
}

.node-header {
  padding: 0.5rem;
  background: #333;
  border-bottom: 1px solid #444;
  border-radius: 6px 6px 0 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.node-content {
  padding: 0.5rem;
}

.node-port {
  padding: 0.25rem 0.5rem;
  margin: 0.25rem;
  background: #444;
  border-radius: 4px;
  cursor: pointer;
}

.node-port:hover {
  background: #555;
}

.input-port {
  margin-left: -0.5rem;
}

.output-port {
  margin-right: -0.5rem;
  text-align: right;
}

.connections {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.connection {
  fill: none;
  stroke: #666;
  stroke-width: 2px;
}

.connection.active {
  stroke: #0a84ff;
}

.connection.preview {
  stroke: #0a84ff;
  stroke-dasharray: 4;
}

.properties-panel {
  width: 300px;
  background: #252526;
  border-left: 1px solid #333;
  overflow-y: auto;
}

.panel-header {
  padding: 1rem;
  background: #333;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.panel-content {
  padding: 1rem;
}

.delete-btn {
  background: none;
  border: none;
  color: #999;
  cursor: pointer;
  font-size: 1.2rem;
  padding: 0 0.5rem;
}

.delete-btn:hover {
  color: #fff;
}
</style>`
