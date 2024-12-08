`<template>
  <div class="run-monitor">
    <!-- Agent Status -->
    <div class="status-panel">
      <h3 class="panel-title">Agent Status</h3>
      <div class="status-grid">
        <div v-for="agent in activeAgents" 
             :key="agent.id" 
             class="status-card"
             :class="getStatusClass(agent.status)">
          <div class="card-header">
            <span class="agent-name">{{ agent.name }}</span>
            <span class="agent-type">{{ agent.type }}</span>
          </div>
          <div class="metrics">
            <div class="metric">
              <span>Tokens:</span>
              <span>{{ agent.metrics.tokens }}</span>
            </div>
            <div class="metric">
              <span>Latency:</span>
              <span>{{ agent.metrics.latency }}ms</span>
            </div>
            <div class="metric">
              <span>Memory:</span>
              <span>{{ formatBytes(agent.metrics.memory) }}</span>
            </div>
          </div>
          <div class="progress-bar" v-if="agent.progress">
            <div class="progress" :style="{ width: agent.progress + '%' }"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Execution Log -->
    <div class="log-panel">
      <h3 class="panel-title">Execution Log</h3>
      <div class="log-container" ref="logContainer">
        <div v-for="log in executionLogs" 
             :key="log.id" 
             class="log-entry"
             :class="getLogClass(log.level)">
          <span class="log-time">{{ formatTime(log.timestamp) }}</span>
          <span class="log-agent">{{ log.agentName }}</span>
          <span class="log-message">{{ log.message }}</span>
        </div>
      </div>
    </div>

    <!-- Performance Metrics -->
    <div class="metrics-panel">
      <h3 class="panel-title">Performance Metrics</h3>
      <div class="metrics-grid">
        <div class="metric-card">
          <h4>Response Time</h4>
          <div id="responseTimeChart" class="chart"></div>
        </div>
        <div class="metric-card">
          <h4>Token Usage</h4>
          <div id="tokenUsageChart" class="chart"></div>
        </div>
        <div class="metric-card">
          <h4>Memory Usage</h4>
          <div id="memoryUsageChart" class="chart"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { defineComponent, ref, onMounted, onUnmounted } from 'vue'
import Plotly from 'plotly.js-dist'

export default defineComponent({
  name: 'RunMonitor',
  
  props: {
    agentId: {
      type: String,
      required: true
    }
  },
  
  setup(props) {
    const activeAgents = ref([])
    const executionLogs = ref([])
    const ws = ref(null)
    const charts = ref({})
    
    // WebSocket connection
    const initWebSocket = () => {
      ws.value = new WebSocket(`ws://localhost:8001/monitor/${props.agentId}`)
      ws.value.onmessage = handleMessage
      ws.value.onclose = () => setTimeout(initWebSocket, 5000)
    }
    
    const handleMessage = (event) => {
      const data = JSON.parse(event.data)
      
      switch (data.type) {
        case 'agent_status':
          updateAgentStatus(data.agent)
          break
        case 'log':
          addLog(data.log)
          break
        case 'metrics':
          updateMetrics(data.metrics)
          break
      }
    }
    
    const updateAgentStatus = (agent) => {
      const index = activeAgents.value.findIndex(a => a.id === agent.id)
      if (index >= 0) {
        activeAgents.value[index] = { ...activeAgents.value[index], ...agent }
      } else {
        activeAgents.value.push(agent)
      }
    }
    
    const addLog = (log) => {
      executionLogs.value.push({
        id: Date.now(),
        timestamp: new Date(),
        ...log
      })
      
      // Keep only last 1000 logs
      if (executionLogs.value.length > 1000) {
        executionLogs.value.shift()
      }
    }
    
    const updateMetrics = (metrics) => {
      // Update response time chart
      Plotly.extendTraces('responseTimeChart', {
        y: [[metrics.latency]]
      }, [0])
      
      // Update token usage chart
      Plotly.extendTraces('tokenUsageChart', {
        y: [[metrics.tokens]]
      }, [0])
      
      // Update memory usage chart
      Plotly.extendTraces('memoryUsageChart', {
        y: [[metrics.memory]]
      }, [0])
    }
    
    const initCharts = () => {
      const layout = {
        showlegend: false,
        margin: { t: 0, r: 0, l: 30, b: 20 },
        height: 100
      }
      
      // Response time chart
      Plotly.newPlot('responseTimeChart', [{
        y: [],
        type: 'line',
        line: { color: '#2196f3' }
      }], layout)
      
      // Token usage chart
      Plotly.newPlot('tokenUsageChart', [{
        y: [],
        type: 'line',
        line: { color: '#4caf50' }
      }], layout)
      
      // Memory usage chart
      Plotly.newPlot('memoryUsageChart', [{
        y: [],
        type: 'line',
        line: { color: '#ff9800' }
      }], layout)
    }
    
    // Utility functions
    const formatTime = (timestamp) => {
      return new Date(timestamp).toLocaleTimeString()
    }
    
    const formatBytes = (bytes) => {
      if (bytes === 0) return '0 B'
      const k = 1024
      const sizes = ['B', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    }
    
    const getStatusClass = (status) => ({
      'status-running': status === 'running',
      'status-completed': status === 'completed',
      'status-error': status === 'error'
    })
    
    const getLogClass = (level) => ({
      'log-info': level === 'info',
      'log-warning': level === 'warning',
      'log-error': level === 'error'
    })
    
    // Lifecycle hooks
    onMounted(() => {
      initWebSocket()
      initCharts()
    })
    
    onUnmounted(() => {
      if (ws.value) ws.value.close()
    })
    
    return {
      activeAgents,
      executionLogs,
      formatTime,
      formatBytes,
      getStatusClass,
      getLogClass
    }
  }
})
</script>

<style scoped>
.run-monitor {
  padding: 1rem;
  background: #1e1e1e;
  color: #fff;
}

.panel-title {
  font-size: 1.1rem;
  font-weight: 500;
  margin-bottom: 1rem;
  color: #ccc;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.status-card {
  background: #2d2d2d;
  border-radius: 8px;
  padding: 1rem;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.agent-name {
  font-weight: 500;
}

.agent-type {
  font-size: 0.9rem;
  color: #999;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.5rem;
  margin: 0.5rem 0;
}

.metric {
  font-size: 0.9rem;
}

.metric span:first-child {
  color: #999;
  margin-right: 0.5rem;
}

.progress-bar {
  height: 4px;
  background: #444;
  border-radius: 2px;
  overflow: hidden;
  margin-top: 0.5rem;
}

.progress {
  height: 100%;
  background: #2196f3;
  transition: width 0.3s ease;
}

.log-panel {
  margin-bottom: 2rem;
}

.log-container {
  height: 300px;
  overflow-y: auto;
  background: #2d2d2d;
  border-radius: 8px;
  padding: 1rem;
}

.log-entry {
  font-family: monospace;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}

.log-time {
  color: #999;
  margin-right: 1rem;
}

.log-agent {
  color: #2196f3;
  margin-right: 1rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
}

.metric-card {
  background: #2d2d2d;
  border-radius: 8px;
  padding: 1rem;
}

.metric-card h4 {
  font-size: 0.9rem;
  color: #ccc;
  margin-bottom: 0.5rem;
}

.chart {
  height: 100px;
}

.status-running {
  border-left: 4px solid #2196f3;
}

.status-completed {
  border-left: 4px solid #4caf50;
}

.status-error {
  border-left: 4px solid #f44336;
}

.log-info {
  color: #fff;
}

.log-warning {
  color: #ff9800;
}

.log-error {
  color: #f44336;
}
</style>`
