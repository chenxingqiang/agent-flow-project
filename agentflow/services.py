"""Services module for AgentFlow."""

import logging
from typing import Dict, Any, Optional, Type, List
from abc import ABC, abstractmethod

class ServiceProvider(ABC):
    """Base class for service providers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize service provider."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service provider."""
        pass
        
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service provider."""
        pass
        
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        pass

class ServiceRegistry:
    """Registry for managing service providers."""
    
    def __init__(self):
        """Initialize service registry."""
        self.logger = logging.getLogger(__name__)
        self._providers: Dict[str, ServiceProvider] = {}
        
    def register_provider(self, name: str, provider: ServiceProvider) -> None:
        """Register a service provider."""
        if name in self._providers:
            self.logger.warning(f"Overwriting existing provider: {name}")
        self._providers[name] = provider
        self.logger.info(f"Registered provider: {name}")
        
    def unregister_provider(self, name: str) -> None:
        """Unregister a service provider."""
        if name in self._providers:
            del self._providers[name]
            self.logger.info(f"Unregistered provider: {name}")
        
    def get_provider(self, name: str) -> Optional[ServiceProvider]:
        """Get a service provider by name."""
        return self._providers.get(name)
        
    def list_providers(self) -> List[str]:
        """List all registered providers."""
        return list(self._providers.keys())
        
    async def initialize_all(self) -> None:
        """Initialize all service providers."""
        for name, provider in self._providers.items():
            try:
                await provider.initialize()
                self.logger.info(f"Initialized provider: {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize provider {name}: {e}")
                raise
        
    async def shutdown_all(self) -> None:
        """Shutdown all service providers."""
        for name, provider in self._providers.items():
            try:
                await provider.shutdown()
                self.logger.info(f"Shutdown provider: {name}")
            except Exception as e:
                self.logger.error(f"Failed to shutdown provider {name}: {e}")
                
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all service providers."""
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception as e:
                self.logger.error(f"Health check failed for provider {name}: {e}")
                results[name] = False
        return results 