`<template>
  <div class="agent-node">
    <!-- Configuration Section -->
    <div class="config-section">
      <div class="form-group">
        <label>Agent Type</label>
        <select v-model="config.type" @change="updateConfig">
          <option value="llm">LLM Agent</option>
          <option value="search">Search Agent</option>
          <option value="tool">Tool Agent</option>
        </select>
      </div>
      
      <div class="form-group">
        <label>Model</label>
        <select v-model="config.model" @change="updateConfig">
          <option value="gpt-4">GPT-4</option>
          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          <option value="claude-2">Claude 2</option>
        </select>
      </div>
      
      <div class="form-group">
        <label>System Prompt</label>
        <textarea v-model="config.systemPrompt" 
                  @change="updateConfig"
                  placeholder="Enter system prompt..."></textarea>
      </div>
    </div>

    <!-- Parameters Section -->
    <div class="params-section">
      <div class="form-group">
        <label>Temperature</label>
        <input type="range" 
               v-model.number="config.temperature"
               min="0" 
               max="2" 
               step="0.1"
               @change="updateConfig">
        <span class="param-value">{{ config.temperature }}</span>
      </div>
      
      <div class="form-group">
        <label>Max Tokens</label>
        <input type="number" 
               v-model.number="config.maxTokens"
               min="1"
               @change="updateConfig">
      </div>
    </div>

    <!-- Tools Section -->
    <div class="tools-section" v-if="config.type === 'tool'">
      <h4>Available Tools</h4>
      <div class="tools-list">
        <div v-for="tool in availableTools" 
             :key="tool.name"
             class="tool-item">
          <input type="checkbox"
                 :value="tool.name"
                 v-model="config.selectedTools"
                 @change="updateConfig">
          <span>{{ tool.label }}</span>
        </div>
      </div>
    </div>

    <!-- Status Section -->
    <div class="status-section">
      <div class="status-indicator" :class="statusClass">
        {{ status }}
      </div>
      <div class="metrics" v-if="metrics">
        <div class="metric">
          <span>Tokens:</span>
          <span>{{ metrics.tokens }}</span>
        </div>
        <div class="metric">
          <span>Latency:</span>
          <span>{{ metrics.latency }}ms</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'AgentNode',
  
  props: {
    initialConfig: {
      type: Object,
      default: () => ({})
    }
  },
  
  data() {
    return {
      config: {
        type: 'llm',
        model: 'gpt-4',
        systemPrompt: '',
        temperature: 0.7,
        maxTokens: 1000,
        selectedTools: []
      },
      status: 'ready',
      metrics: null,
      availableTools: [
        { name: 'web_search', label: 'Web Search' },
        { name: 'calculator', label: 'Calculator' },
        { name: 'code_interpreter', label: 'Code Interpreter' },
        { name: 'file_manager', label: 'File Manager' }
      ]
    }
  },
  
  computed: {
    statusClass() {
      return {
        'status-ready': this.status === 'ready',
        'status-running': this.status === 'running',
        'status-error': this.status === 'error'
      }
    }
  },
  
  created() {
    // Initialize with props
    this.config = { ...this.config, ...this.initialConfig }
  },
  
  methods: {
    updateConfig() {
      this.$emit('update', this.config)
    },
    
    updateStatus(status, metrics = null) {
      this.status = status
      this.metrics = metrics
    }
  }
}
</script>

<style scoped>
.agent-node {
  padding: 1rem;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  color: #ccc;
}

select, input[type="number"], textarea {
  width: 100%;
  padding: 0.5rem;
  background: #333;
  border: 1px solid #444;
  border-radius: 4px;
  color: #fff;
}

textarea {
  min-height: 80px;
  resize: vertical;
}

.param-value {
  margin-left: 0.5rem;
  color: #ccc;
}

.tools-section {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #444;
}

.tools-section h4 {
  margin-bottom: 0.5rem;
  color: #ccc;
}

.tool-item {
  display: flex;
  align-items: center;
  margin-bottom: 0.5rem;
}

.tool-item input[type="checkbox"] {
  margin-right: 0.5rem;
}

.status-section {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #444;
}

.status-indicator {
  display: inline-block;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.9rem;
}

.status-ready {
  background: #2d5a27;
  color: #4caf50;
}

.status-running {
  background: #1a4971;
  color: #2196f3;
}

.status-error {
  background: #712b29;
  color: #f44336;
}

.metrics {
  margin-top: 0.5rem;
  display: flex;
  justify-content: space-between;
}

.metric {
  font-size: 0.9rem;
  color: #ccc;
}

.metric span:first-child {
  margin-right: 0.5rem;
  color: #999;
}
</style>`
