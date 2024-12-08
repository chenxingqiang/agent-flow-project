`<template>
  <div class="chat-container bg-white rounded-lg shadow-lg">
    <!-- Chat Header -->
    <div class="chat-header border-b p-4 flex justify-between items-center">
      <div class="flex items-center">
        <div class="w-3 h-3 rounded-full mr-2" 
             :class="isConnected ? 'bg-green-500' : 'bg-red-500'"></div>
        <h2 class="text-lg font-semibold">{{ selectedAgent ? selectedAgent.name : 'Chat' }}</h2>
      </div>
      <div class="flex items-center space-x-2">
        <select v-model="selectedAgent" 
                class="border rounded px-2 py-1 text-sm">
          <option v-for="agent in agents" 
                  :key="agent.id" 
                  :value="agent">
            {{ agent.name }}
          </option>
        </select>
        <button @click="clearHistory" 
                class="text-gray-500 hover:text-gray-700">
          <i class="fas fa-trash-alt"></i>
        </button>
      </div>
    </div>

    <!-- Chat Messages -->
    <div class="chat-messages p-4 h-[500px] overflow-y-auto" ref="messageContainer">
      <div v-for="message in messages" 
           :key="message.id" 
           class="mb-4">
        <!-- Message Container -->
        <div :class="[
          'max-w-[80%] rounded-lg p-3',
          message.role === 'user' 
            ? 'ml-auto bg-blue-500 text-white' 
            : 'bg-gray-100 text-gray-800'
        ]">
          <!-- Message Header -->
          <div class="flex items-center mb-1">
            <span class="text-sm font-medium">
              {{ message.role === 'user' ? 'You' : selectedAgent?.name }}
            </span>
            <span class="text-xs ml-2 opacity-70">
              {{ formatTime(message.timestamp) }}
            </span>
          </div>

          <!-- Message Content -->
          <div class="message-content">
            <div v-if="message.type === 'text'" 
                 class="whitespace-pre-wrap">
              {{ message.content }}
            </div>
            <div v-else-if="message.type === 'code'" 
                 class="bg-gray-800 text-white p-2 rounded">
              <pre><code>{{ message.content }}</code></pre>
            </div>
            <div v-else-if="message.type === 'image'" 
                 class="mt-2">
              <img :src="message.content" 
                   class="max-w-full rounded" 
                   alt="Message image">
            </div>
          </div>

          <!-- Message Status -->
          <div v-if="message.status" 
               class="text-xs mt-1" 
               :class="getStatusColor(message.status)">
            {{ message.status }}
          </div>
        </div>

        <!-- Thinking Indicator -->
        <div v-if="message.thinking" 
             class="flex items-center mt-2">
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>
    </div>

    <!-- Input Area -->
    <div class="chat-input border-t p-4">
      <div class="flex items-end space-x-2">
        <!-- Text Input -->
        <div class="flex-1">
          <textarea v-model="newMessage" 
                    @keydown.enter.prevent="sendMessage"
                    placeholder="Type your message..."
                    class="w-full border rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows="1"
                    :disabled="!isConnected || isProcessing"></textarea>
        </div>

        <!-- Action Buttons -->
        <div class="flex space-x-2">
          <!-- Upload Button -->
          <button class="p-2 text-gray-500 hover:text-gray-700"
                  :disabled="!isConnected || isProcessing">
            <i class="fas fa-paperclip"></i>
            <input type="file" 
                   class="hidden" 
                   @change="handleFileUpload" 
                   accept="image/*,.pdf,.doc,.docx">
          </button>

          <!-- Send Button -->
          <button @click="sendMessage"
                  class="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
                  :disabled="!isConnected || isProcessing || !newMessage.trim()">
            <i class="fas fa-paper-plane"></i>
          </button>
        </div>
      </div>

      <!-- Input Options -->
      <div class="flex items-center mt-2 space-x-4 text-sm text-gray-500">
        <label class="flex items-center">
          <input type="checkbox" 
                 v-model="settings.streamResponse" 
                 class="mr-1">
          Stream Response
        </label>
        <label class="flex items-center">
          <input type="checkbox" 
                 v-model="settings.enableContext" 
                 class="mr-1">
          Enable Context
        </label>
        <span class="flex-1"></span>
        <span v-if="tokenCount" class="text-xs">
          Tokens: {{ tokenCount }}
        </span>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ChatBox',
  
  data() {
    return {
      messages: [],
      newMessage: '',
      isConnected: false,
      isProcessing: false,
      selectedAgent: null,
      agents: [],
      tokenCount: 0,
      settings: {
        streamResponse: true,
        enableContext: true
      }
    }
  },

  props: {
    initialAgents: {
      type: Array,
      default: () => []
    }
  },

  created() {
    this.agents = this.initialAgents
    if (this.agents.length > 0) {
      this.selectedAgent = this.agents[0]
    }
    this.initWebSocket()
  },

  methods: {
    initWebSocket() {
      this.ws = new WebSocket('ws://localhost:8001/chat')
      this.ws.onopen = this.onWebSocketOpen
      this.ws.onclose = this.onWebSocketClose
      this.ws.onmessage = this.onWebSocketMessage
    },

    onWebSocketOpen() {
      this.isConnected = true
      this.addSystemMessage('Connected to server')
    },

    onWebSocketClose() {
      this.isConnected = false
      this.addSystemMessage('Disconnected from server')
      setTimeout(this.initWebSocket, 5000)
    },

    onWebSocketMessage(event) {
      const data = JSON.parse(event.data)
      
      switch (data.type) {
        case 'response':
          this.handleResponse(data)
          break
        case 'stream':
          this.handleStreamResponse(data)
          break
        case 'error':
          this.handleError(data)
          break
        case 'status':
          this.handleStatusUpdate(data)
          break
      }
    },

    async sendMessage() {
      if (!this.newMessage.trim() || !this.selectedAgent) return

      // Add user message
      const userMessage = {
        id: Date.now(),
        role: 'user',
        content: this.newMessage,
        type: 'text',
        timestamp: new Date(),
        status: 'sent'
      }
      this.messages.push(userMessage)

      // Add thinking indicator
      const thinkingMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        thinking: true,
        timestamp: new Date()
      }
      this.messages.push(thinkingMessage)

      // Send to server
      this.isProcessing = true
      try {
        await this.ws.send(JSON.stringify({
          type: 'message',
          content: this.newMessage,
          agent_id: this.selectedAgent.id,
          settings: this.settings
        }))
        
        this.newMessage = ''
      } catch (error) {
        this.handleError({ error: 'Failed to send message' })
      }
      
      this.scrollToBottom()
    },

    handleResponse(data) {
      // Remove thinking indicator
      this.messages = this.messages.filter(m => !m.thinking)

      // Add response message
      this.messages.push({
        id: Date.now(),
        role: 'assistant',
        content: data.content,
        type: data.content_type || 'text',
        timestamp: new Date(),
        status: 'received'
      })

      this.isProcessing = false
      this.scrollToBottom()
    },

    handleStreamResponse(data) {
      const lastMessage = this.messages[this.messages.length - 1]
      
      if (lastMessage && lastMessage.role === 'assistant') {
        lastMessage.content += data.content
      } else {
        this.messages.push({
          id: Date.now(),
          role: 'assistant',
          content: data.content,
          type: 'text',
          timestamp: new Date(),
          status: 'streaming'
        })
      }
      
      this.scrollToBottom()
    },

    handleError(data) {
      this.messages = this.messages.filter(m => !m.thinking)
      
      this.messages.push({
        id: Date.now(),
        role: 'system',
        content: data.error,
        type: 'text',
        timestamp: new Date(),
        status: 'error'
      })
      
      this.isProcessing = false
      this.scrollToBottom()
    },

    handleStatusUpdate(data) {
      const message = this.messages.find(m => m.id === data.message_id)
      if (message) {
        message.status = data.status
      }
    },

    async handleFileUpload(event) {
      const file = event.target.files[0]
      if (!file) return

      try {
        const formData = new FormData()
        formData.append('file', file)

        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        })

        const data = await response.json()
        
        if (data.url) {
          this.messages.push({
            id: Date.now(),
            role: 'user',
            content: data.url,
            type: 'image',
            timestamp: new Date(),
            status: 'sent'
          })
        }
      } catch (error) {
        this.handleError({ error: 'Failed to upload file' })
      }
    },

    addSystemMessage(content) {
      this.messages.push({
        id: Date.now(),
        role: 'system',
        content,
        type: 'text',
        timestamp: new Date()
      })
    },

    clearHistory() {
      this.messages = []
    },

    formatTime(timestamp) {
      return new Date(timestamp).toLocaleTimeString()
    },

    getStatusColor(status) {
      const colors = {
        'sent': 'text-gray-500',
        'received': 'text-green-500',
        'error': 'text-red-500',
        'streaming': 'text-blue-500'
      }
      return colors[status] || 'text-gray-500'
    },

    scrollToBottom() {
      this.$nextTick(() => {
        const container = this.$refs.messageContainer
        container.scrollTop = container.scrollHeight
      })
    }
  },

  watch: {
    messages: {
      deep: true,
      handler() {
        this.scrollToBottom()
      }
    }
  },

  beforeDestroy() {
    if (this.ws) {
      this.ws.close()
    }
  }
}
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
}

.typing-indicator {
  display: flex;
  align-items: center;
  padding: 4px 8px;
}

.typing-indicator span {
  width: 6px;
  height: 6px;
  background-color: #90cdf4;
  border-radius: 50%;
  margin: 0 2px;
  animation: typing 1s infinite ease-in-out;
}

.typing-indicator span:nth-child(1) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(2) { animation-delay: 0.3s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-4px); }
}

textarea {
  resize: none;
  min-height: 40px;
  max-height: 120px;
}

.message-content pre {
  overflow-x: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
}
</style>`
