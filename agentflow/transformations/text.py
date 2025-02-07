from .base import TransformationStrategy

class TextTransformationStrategy(TransformationStrategy):
    """Strategy for text transformations."""
    
    def transform(self, text):
        """Transform text data.
        
        Args:
            text: Text to transform
            
        Returns:
            Transformed text
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
            
        # Apply configured transformations
        transformed = text
        
        # Lowercase if configured
        if self.config.get('lowercase', False):
            transformed = transformed.lower()
            
        # Strip whitespace if configured
        if self.config.get('strip', False):
            transformed = transformed.strip()
            
        # Replace patterns if configured
        replacements = self.config.get('replacements', {})
        for old, new in replacements.items():
            transformed = transformed.replace(old, new)
            
        return transformed 