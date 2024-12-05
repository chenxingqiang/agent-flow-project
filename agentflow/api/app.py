from fastapi import FastAPI, HTTPException
from .workflow_server import app as workflow_app

app = FastAPI(title="AgentFlow API")

# Mount workflow API
app.mount("/workflow", workflow_app)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions"""
    return {
        "status_code": 400,
        "detail": str(exc)
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return {
        "status_code": 500,
        "detail": "Internal server error"
    } 