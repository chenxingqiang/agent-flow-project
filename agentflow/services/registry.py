"""Service registry module for AgentFlow."""

from typing import Dict, Any, Optional, Type, List
from abc import ABC, abstractmethod
import logging

class ServiceProvider(ABC):
    """Abstract base class for service providers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize service provider."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        pass
    
    @abstractmethod
    async def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute service request."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up service resources."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get service configuration."""
        return self.config
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update service configuration."""
        self.config.update(config)

class ServiceRegistry:
    """Registry for managing service providers."""
    
    def __init__(self):
        """Initialize service registry."""
        self._providers: Dict[str, ServiceProvider] = {}
        self.logger = logging.getLogger(__name__)
    
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
        """Initialize all registered providers."""
        for name, provider in self._providers.items():
            try:
                await provider.initialize()
                self.logger.info(f"Initialized provider: {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize provider {name}: {e}")
    
    async def cleanup_all(self) -> None:
        """Clean up all registered providers."""
        for name, provider in self._providers.items():
            try:
                await provider.cleanup()
                self.logger.info(f"Cleaned up provider: {name}")
            except Exception as e:
                self.logger.error(f"Failed to clean up provider {name}: {e}")

class ServiceFactory:
    """Factory for creating service providers."""
    
    _providers: Dict[str, Type[ServiceProvider]] = {}
    
    @classmethod
    def register_provider_class(cls, name: str, provider_class: Type[ServiceProvider]) -> None:
        """Register a provider class."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create_provider(cls, name: str, config: Optional[Dict[str, Any]] = None) -> ServiceProvider:
        """Create a provider instance."""
        if name not in cls._providers:
            raise ValueError(f"Unknown provider type: {name}")
        
        provider_class = cls._providers[name]
        return provider_class(config) 