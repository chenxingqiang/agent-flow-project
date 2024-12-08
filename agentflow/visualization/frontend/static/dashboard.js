const app = Vue.createApp({
    data() {
        return {
            isConnected: false,
            connectionStatus: 'Disconnected',
            agents: [],
            logs: [],
            metrics: {
                cpu: [],
                memory: [],
                ray: []
            }
        }
    },

    methods: {
        // WebSocket Connection
        initWebSocket() {
            this.ws = new WebSocket('ws://localhost:8001/live');
            this.ws.onopen = this.onWebSocketOpen;
            this.ws.onclose = this.onWebSocketClose;
            this.ws.onmessage = this.onWebSocketMessage;
        },

        onWebSocketOpen() {
            this.isConnected = true;
            this.connectionStatus = 'Connected';
            this.addLog('info', 'Connected to server');
        },

        onWebSocketClose() {
            this.isConnected = false;
            this.connectionStatus = 'Disconnected';
            this.addLog('error', 'Disconnected from server');
            // Try to reconnect
            setTimeout(this.initWebSocket, 5000);
        },

        onWebSocketMessage(event) {
            const data = JSON.parse(event.data);
            switch (data.type) {
                case 'agent_status':
                    this.updateAgentStatus(data.data);
                    break;
                case 'metrics':
                    this.updateMetrics(data.data);
                    break;
                case 'log':
                    this.addLog(data.data.level, data.data.message);
                    break;
                case 'workflow':
                    this.updateWorkflowVisualization(data.data);
                    break;
            }
        },

        // Status Management
        updateAgentStatus(agentData) {
            const index = this.agents.findIndex(a => a.id === agentData.id);
            if (index >= 0) {
                this.agents[index] = { ...this.agents[index], ...agentData };
            } else {
                this.agents.push(agentData);
            }
        },

        getStatusColor(status) {
            const colors = {
                'running': 'text-green-600',
                'waiting': 'text-yellow-600',
                'error': 'text-red-600',
                'completed': 'text-blue-600'
            };
            return colors[status] || 'text-gray-600';
        },

        // Logging
        addLog(level, message) {
            this.logs.unshift({
                id: Date.now(),
                timestamp: new Date().toISOString(),
                level,
                message
            });
            if (this.logs.length > 1000) {
                this.logs.pop();
            }
        },

        getLogLevelColor(level) {
            const colors = {
                'info': 'text-gray-600',
                'warning': 'text-yellow-600',
                'error': 'text-red-600',
                'success': 'text-green-600'
            };
            return colors[level] || 'text-gray-600';
        },

        // Metrics Visualization
        initCharts() {
            this.initRayMetricsChart();
            this.initResourceCharts();
        },

        initRayMetricsChart() {
            const layout = {
                title: 'Ray Cluster Metrics',
                showlegend: true,
                height: 250,
                margin: { t: 30, r: 20, l: 40, b: 30 }
            };

            Plotly.newPlot('rayMetricsChart', [{
                y: [],
                type: 'line',
                name: 'Tasks'
            }], layout);
        },

        initResourceCharts() {
            const layout = {
                showlegend: false,
                height: 120,
                margin: { t: 10, r: 10, l: 30, b: 20 }
            };

            Plotly.newPlot('cpuChart', [{
                y: [],
                type: 'line',
                fill: 'tozeroy'
            }], layout);

            Plotly.newPlot('memoryChart', [{
                y: [],
                type: 'line',
                fill: 'tozeroy'
            }], layout);
        },

        updateMetrics(metricsData) {
            // Update Ray metrics
            Plotly.extendTraces('rayMetricsChart', {
                y: [[metricsData.ray.tasks]]
            }, [0]);

            // Update resource metrics
            Plotly.extendTraces('cpuChart', {
                y: [[metricsData.cpu]]
            }, [0]);

            Plotly.extendTraces('memoryChart', {
                y: [[metricsData.memory]]
            }, [0]);

            // Keep only last 50 points
            if (this.metrics.ray.length > 50) {
                Plotly.relayout('rayMetricsChart', {
                    xaxis: {
                        range: [this.metrics.ray.length - 50, this.metrics.ray.length]
                    }
                });
            }
        },

        // Workflow Visualization
        initWorkflowVisualization() {
            this.ellStudio = new EllStudio({
                container: 'workflowVisualization',
                apiKey: 'your_ell_studio_api_key'
            });
        },

        updateWorkflowVisualization(workflowData) {
            this.ellStudio.updateGraph(workflowData);
        }
    },

    mounted() {
        this.initWebSocket();
        this.initCharts();
        this.initWorkflowVisualization();

        // Add some initial test data
        this.addLog('info', 'Dashboard initialized');
        this.updateAgentStatus({
            id: 1,
            name: 'Research Agent',
            status: 'running'
        });
    }
});

app.mount('#app');
