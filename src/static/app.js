// Global app utilities and shared functionality

// Utility functions
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function formatRuntime(milliseconds) {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    
    if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    } else {
        return `${seconds}s`;
    }
}

function formatOptimality(optimality) {
    return (optimality * 100).toFixed(1) + '%';
}

// WebSocket connection management
class WebSocketManager {
    constructor(clientId, handlers = {}) {
        this.clientId = clientId;
        this.handlers = handlers;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000;
    }
    
    connect() {
        try {
            this.ws = new WebSocket(`ws://localhost:8001/ws/${this.clientId}`);
            
            this.ws.onopen = (event) => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                if (this.handlers.onOpen) {
                    this.handlers.onOpen(event);
                }
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    if (this.handlers.onMessage) {
                        this.handlers.onMessage(message);
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.ws.onclose = (event) => {
                console.log('WebSocket disconnected');
                if (this.handlers.onClose) {
                    this.handlers.onClose(event);
                }
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                if (this.handlers.onError) {
                    this.handlers.onError(error);
                }
            };
        } catch (error) {
            console.error('Error creating WebSocket:', error);
            this.attemptReconnect();
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => this.connect(), this.reconnectDelay);
        } else {
            console.error('Max reconnection attempts reached');
            if (this.handlers.onMaxReconnectAttemptsReached) {
                this.handlers.onMaxReconnectAttemptsReached();
            }
        }
    }
    
    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
            return true;
        } else {
            console.warn('WebSocket not connected');
            return false;
        }
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// API client for REST endpoints
class APIClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }
    
    async get(endpoint) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Error fetching ${endpoint}:`, error);
            throw error;
        }
    }
    
    async post(endpoint, data) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Error posting to ${endpoint}:`, error);
            throw error;
        }
    }
}

// Notification system
class NotificationManager {
    constructor() {
        this.container = this.createContainer();
    }
    
    createContainer() {
        const container = document.createElement('div');
        container.className = 'notification-container';
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            max-width: 400px;
        `;
        document.body.appendChild(container);
        return container;
    }
    
    show(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            background: ${this.getBackgroundColor(type)};
            color: white;
            padding: 1rem;
            margin-bottom: 0.5rem;
            border-radius: 6px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            animation: slideIn 0.3s ease;
        `;
        notification.textContent = message;
        
        this.container.appendChild(notification);
        
        if (duration > 0) {
            setTimeout(() => {
                this.remove(notification);
            }, duration);
        }
        
        return notification;
    }
    
    getBackgroundColor(type) {
        const colors = {
            info: '#3498db',
            success: '#27ae60',
            warning: '#f39c12',
            error: '#e74c3c'
        };
        return colors[type] || colors.info;
    }
    
    remove(notification) {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Global instances
window.apiClient = new APIClient();
window.notifications = new NotificationManager();

// Page-specific initialization
document.addEventListener('DOMContentLoaded', function() {
    // Add active class to current page nav link
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.style.color = '#2c3e50';
            link.style.fontWeight = '600';
        }
    });
    
    // Handle any global initialization
    console.log('SciPathBench Web UI initialized');
});